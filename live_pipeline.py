from __future__ import annotations

import html, json, os, re, sys
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mlb_model.features import build_features
from mlb_model.market_intelligence_v2_9 import MarketIntelligenceV29Config, run_v29
from mlb_model.moneyline_classifier_v2_4 import MoneylineV24Config, build_candidate_frame, fit_models, score_candidates
from mlb_model.odds import canonical_team, fetch_the_odds_api_json, normalize_the_odds_api_json
from mlb_model.predict import predict_scores
from run_predictions import build_card, write_markdown

ASSETS = ROOT / "assets"
WORK = ROOT / "output" / "live_work"

def fetch_json(url: str) -> dict:
    with urlopen(Request(url, headers={"User-Agent":"mlb-market-intelligence-v2.9"}), timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))

def team(game: dict, side: str) -> dict:
    return ((game.get("teams") or {}).get(side) or {}).get("team") or {}

def pitcher(game: dict, side: str) -> str:
    return str((((game.get("teams") or {}).get(side) or {}).get("probablePitcher") or {}).get("fullName") or "")

def schedule_frame(payload: dict) -> pd.DataFrame:
    rows=[]
    for block in payload.get("dates") or []:
        for game in block.get("games") or []:
            h,a=team(game,"home"),team(game,"away"); ht=(game.get("teams") or {}).get("home") or {}; at=(game.get("teams") or {}).get("away") or {}
            rows.append({"season":int(str(block.get("date"))[:4]),"game_date":str(block.get("date")),"game_pk":int(game["gamePk"]),"game_type":game.get("gameType","R"),"home_team":canonical_team(h.get("name")),"away_team":canonical_team(a.get("name")),"home_team_id":int(h.get("id") or 0),"away_team_id":int(a.get("id") or 0),"venue_id":int((game.get("venue") or {}).get("id") or 0),"home_score":int(ht.get("score") or 0),"away_score":int(at.get("score") or 0),"home_probable_pitcher":pitcher(game,"home"),"away_probable_pitcher":pitcher(game,"away"),"game_start_time_utc":game.get("gameDate"),"game_status":((game.get("status") or {}).get("detailedState") or ""),"temperature_f":70.0,"wind_mph":0.0,"humidity_pct":np.nan,"weather_condition":"Neutral","wind_direction":"","weather_source":"neutral_placeholder","weather_join_matched":False})
    frame=pd.DataFrame(rows)
    if not frame.empty:
        frame["total_runs"]=frame.home_score+frame.away_score; frame["home_run_diff"]=frame.home_score-frame.away_score
    return frame

def next_slate() -> tuple[str,pd.DataFrame]:
    today=date.today(); end=today+timedelta(days=7)
    url="https://statsapi.mlb.com/api/v1/schedule?"+urlencode({"sportId":1,"startDate":today.isoformat(),"endDate":end.isoformat(),"hydrate":"probablePitcher,team,venue"})
    frame=schedule_frame(fetch_json(url))
    if frame.empty: raise RuntimeError("MLB Stats API returned no games in the next seven days.")
    active=frame[~frame.game_status.str.contains("Final|Completed",case=False,na=False)]
    if active.empty: raise RuntimeError("No upcoming MLB games were found in the next seven days.")
    day=sorted(active.game_date.unique())[0]
    return day, active[active.game_date.eq(day)].copy()

def refresh_games(target: str, slate: pd.DataFrame) -> pd.DataFrame:
    history=pd.read_csv(ASSETS/"data"/"raw"/"games_history.csv",low_memory=False)
    start=(pd.to_datetime(history.game_date).max()+pd.Timedelta(days=1)).date().isoformat()
    end=(pd.to_datetime(target)-pd.Timedelta(days=1)).date().isoformat()
    additions=pd.DataFrame()
    if start<=end:
        url="https://statsapi.mlb.com/api/v1/schedule?"+urlencode({"sportId":1,"startDate":start,"endDate":end,"hydrate":"probablePitcher,team,venue"})
        additions=schedule_frame(fetch_json(url))
        if not additions.empty: additions=additions[additions.game_status.str.contains("Final|Completed",case=False,na=False)]
    return pd.concat([history,additions,slate],ignore_index=True,sort=False).drop_duplicates("game_pk",keep="last").sort_values(["game_date","game_pk"])

def key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+","",str(canonical_team(value)).lower())

def attach_odds(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    p=predictions.copy(); o=odds.copy(); p["_key"]=p.game_date.astype(str)+"|"+p.away_team.map(key)+"|"+p.home_team.map(key); o["_key"]=o.game_date.astype(str)+"|"+o.away_team.map(key)+"|"+o.home_team.map(key)
    cols=["_key","sportsbook","sportsbook_count","commence_time_utc","market_timestamp_utc","away_moneyline","home_moneyline","total_line","total_price_over","total_price_under"]
    p=p.merge(o[[c for c in cols if c in o]],on="_key",how="left").drop(columns="_key")
    for c in ["away_moneyline","home_moneyline","total_line","total_price_over","total_price_under"]: p[c]=pd.to_numeric(p.get(c),errors="coerce")
    p["total_line_open"]=p.total_line; p["total_price_over_open"]=p.total_price_over; p["total_price_under_open"]=p.total_price_under; p["home_moneyline_open"]=p.home_moneyline; p["away_moneyline_open"]=p.away_moneyline; p["total_line_move"]=0.0; p["total_line_source"]="the_odds_api_current_as_open"; p["odds_timestamp"]=p.get("market_timestamp_utc"); p["market_open_timestamp_utc"]=p.get("market_timestamp_utc"); p["game_date_odds"]=p.game_date
    return p

def _number(value: object) -> float:
    return pd.to_numeric(pd.Series([value]),errors="coerce").iloc[0]

def _american(value: object) -> str:
    value=_number(value)
    return "" if pd.isna(value) else f"{value:+.0f}"

def build_full_slate(outputs: dict[str,pd.DataFrame], target: str) -> pd.DataFrame:
    ml=outputs.get("ml_scored_candidates",pd.DataFrame()).copy(); ou=outputs.get("ou_scored_candidates",pd.DataFrame()).copy()
    ml=ml[ml.game_date.astype(str).eq(target)] if not ml.empty else ml; ou=ou[ou.game_date.astype(str).eq(target)] if not ou.empty else ou
    ml_orders=outputs.get("ml_orders",pd.DataFrame()); ou_orders=outputs.get("ou_orders",pd.DataFrame())
    ml_edges=set(zip(ml_orders.get("game_pk",pd.Series(dtype=object)).astype(str),ml_orders.get("side",pd.Series(dtype=object)).astype(str)))
    ou_edges=set(zip(ou_orders.get("game_pk",pd.Series(dtype=object)).astype(str),ou_orders.get("side",pd.Series(dtype=object)).astype(str)))
    ml_best={}
    if not ml.empty:
        probability="ml_game_probability" if "ml_game_probability" in ml else "model_probability"
        for game_pk,group in ml.groupby("game_pk"):
            ranked=group.assign(_p=pd.to_numeric(group[probability],errors="coerce")).sort_values(["_p","rank_score"],ascending=[False,False],na_position="last")
            ml_best[str(game_pk)]=ranked.iloc[0]
    ou_best={}
    if not ou.empty:
        for game_pk,group in ou.groupby("game_pk"):
            ranked=group.assign(_p=pd.to_numeric(group.get("ou_game_probability"),errors="coerce")).sort_values(["_p","rank_score"],ascending=[False,False],na_position="last")
            ou_best[str(game_pk)]=ranked.iloc[0]
    rows=[]
    for game_pk in sorted(set(ml_best)|set(ou_best)):
        m=ml_best.get(game_pk); o=ou_best.get(game_pk); base=m if m is not None else o
        ml_side=str(m.get("side","")) if m is not None else ""; ou_side=str(o.get("side","")) if o is not None else ""
        winner=(m.get("display_side") if m is not None else "") or ((m.get("home_team") if ml_side=="HOME" else m.get("away_team")) if m is not None else "")
        total=_number(o.get("opening_total",o.get("total_line",np.nan))) if o is not None else np.nan
        model_total=_number(o.get("model_total",np.nan)) if o is not None else np.nan
        edge_labels=[]
        if (game_pk,ml_side) in ml_edges: edge_labels.append(f"ML: {winner}")
        if (game_pk,ou_side) in ou_edges: edge_labels.append(f"OU: {ou_side} {total:g}")
        rows.append({"date":target,"game":f"{base.get('away_team','')} @ {base.get('home_team','')}","predicted_winner":winner,"moneyline":_american(m.get("selected_price",m.get("wager_price",np.nan))) if m is not None else "","winner_probability":_number(m.get("ml_game_probability",m.get("model_probability",np.nan))) if m is not None else np.nan,"ou_prediction":f"{ou_side} {total:g}" if o is not None and pd.notna(total) else ou_side,"model_total":model_total,"ou_probability":_number(o.get("ou_game_probability",np.nan)) if o is not None else np.nan,"edge_bet":" | ".join(edge_labels),"has_edge":bool(edge_labels)})
    return pd.DataFrame(rows)

def write_full_slate(frame: pd.DataFrame, target: str, path: Path) -> None:
    lines=[f"# MLB v2.9 full-slate predictions — {target}","","🟩 **EDGE** marks selections that passed the v2.9 edge filters.",""]
    if frame.empty:
        lines.append("No upcoming games were available.")
    else:
        view=frame.copy(); view["winner_probability"]=pd.to_numeric(view.winner_probability,errors="coerce").map(lambda x:f"{x:.1%}" if pd.notna(x) else ""); view["ou_probability"]=pd.to_numeric(view.ou_probability,errors="coerce").map(lambda x:f"{x:.1%}" if pd.notna(x) else ""); view["model_total"]=pd.to_numeric(view.model_total,errors="coerce").map(lambda x:f"{x:.1f}" if pd.notna(x) else ""); view["edge_bet"]=view.apply(lambda r:f"🟩 **EDGE — {r.edge_bet}**" if r.has_edge else "—",axis=1)
        view=view.rename(columns={"game":"Game","predicted_winner":"Predicted winner","moneyline":"ML","winner_probability":"Win probability","ou_prediction":"O/U prediction","model_total":"Model total","ou_probability":"O/U probability","edge_bet":"Edge bets"})
        lines.append(view[["Game","Predicted winner","ML","Win probability","O/U prediction","Model total","O/U probability","Edge bets"]].to_markdown(index=False))
    lines.extend(["","Research output only; verify line freshness before acting.",""]); path.write_text("\n".join(lines),encoding="utf-8")

def write_dashboard(frame: pd.DataFrame, target: str, site: Path) -> None:
    site.mkdir(parents=True,exist_ok=True)
    rows=[]
    for _,r in frame.iterrows():
        edge=bool(r.get("has_edge",False)); probability=_number(r.get("winner_probability")); ou_probability=_number(r.get("ou_probability")); model_total=_number(r.get("model_total"))
        values=[r.get("game",""),r.get("predicted_winner",""),r.get("moneyline",""),f"{probability:.1%}" if pd.notna(probability) else "",r.get("ou_prediction",""),f"{model_total:.1f}" if pd.notna(model_total) else "",f"{ou_probability:.1%}" if pd.notna(ou_probability) else "",f"EDGE — {r.get('edge_bet','')}" if edge else "—"]
        cells="".join(f"<td>{html.escape(str(v))}</td>" for v in values); rows.append(f"<tr class={'edge' if edge else 'standard'}>{cells}</tr>")
    body="".join(rows) if rows else '<tr><td colspan="8" class="empty">No upcoming games were available.</td></tr>'
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MLB v2.9 Predictions</title><style>
body{{margin:0;background:#09111f;color:#e8eef8;font-family:Inter,Segoe UI,Arial,sans-serif}}.wrap{{max-width:1400px;margin:0 auto;padding:32px 20px}}.hero{{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:22px}}h1{{margin:0;font-size:30px}}.date{{color:#9fb0ca;margin-top:6px}}.badge{{background:#17345e;color:#9dccff;padding:8px 12px;border-radius:999px;font-weight:700}}.card{{background:#111d2f;border:1px solid #263752;border-radius:14px;overflow:auto;box-shadow:0 14px 40px #0005}}table{{width:100%;border-collapse:collapse;min-width:980px}}th,td{{padding:13px 14px;text-align:left;border-bottom:1px solid #25344d}}th{{position:sticky;top:0;background:#17253a;color:#b9c8dc;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}tr.edge{{background:#123b2b}}tr.edge td:last-child{{color:#72efa9;font-weight:800}}tr.standard:hover{{background:#15243a}}.footer{{display:flex;justify-content:space-between;gap:16px;color:#8495ae;font-size:13px;margin-top:16px}}a{{color:#77b7ff}}.empty{{text-align:center;padding:40px}}@media(max-width:700px){{.hero,.footer{{display:block}}.badge{{display:inline-block;margin-top:12px}}}}
</style></head><body><main class="wrap"><div class="hero"><div><h1>MLB Market Intelligence v2.9</h1><div class="date">Full-slate predictions for {html.escape(target)}</div></div><div class="badge">Green rows = qualifying edge bets</div></div><div class="card"><table><thead><tr><th>Game</th><th>Predicted winner</th><th>ML</th><th>Win probability</th><th>O/U prediction</th><th>Model total</th><th>O/U probability</th><th>Edge bets</th></tr></thead><tbody>{body}</tbody></table></div><div class="footer"><span>Research output only. Verify sportsbook line freshness.</span><a href="all_game_predictions.csv">Download predictions CSV</a></div></main></body></html>'''
    (site/"index.html").write_text(page,encoding="utf-8")
    frame.to_csv(site/"all_game_predictions.csv",index=False)

def main() -> None:
    api_key=os.environ.get("THE_ODDS_API_KEY")
    if not api_key: raise SystemExit("THE_ODDS_API_KEY is not set in this Windows account.")
    WORK.mkdir(parents=True,exist_ok=True)
    target,slate=next_slate(); games=refresh_games(target,slate); features=build_features(games); slate_features=features[features.game_date.astype(str).eq(target)].copy()
    predictions=predict_scores(slate_features,ASSETS/"models"/"score_model")
    context=["game_pk","home_probable_pitcher","away_probable_pitcher","home_sp_xFIP","away_sp_xFIP","home_sp_kbb","away_sp_kbb","home_wRC_plus_vs_hand","away_wRC_plus_vs_hand","bullpen_xFIP_diff","bullpen_fatigue_diff","park_run_factor","weather_run_index"]
    predictions=predictions.merge(slate_features[[c for c in context if c in slate_features]],on="game_pk",how="left")
    payload=fetch_the_odds_api_json(api_key,regions="us",markets="h2h,totals",odds_format="american"); odds=normalize_the_odds_api_json(payload,sportsbook="consensus"); odds=odds[odds.game_date.astype(str).eq(target)]
    if odds.empty: raise RuntimeError(f"The Odds API returned no MLB lines for {target}.")
    current=attach_odds(predictions,odds)
    training_predictions=pd.read_csv(ASSETS/"data"/"processed"/"training_predictions.csv",low_memory=False)
    combined_predictions=pd.concat([training_predictions,current],ignore_index=True,sort=False).drop_duplicates("game_pk",keep="last")
    historical_features=pd.read_csv(ASSETS/"data"/"processed"/"training_features.csv",low_memory=False)
    combined_features=pd.concat([historical_features,features[features.game_date.astype(str).gt(str(historical_features.game_date.max()))]],ignore_index=True,sort=False).drop_duplicates("game_pk",keep="last")
    ml_cfg=MoneylineV24Config(train_start_season=2021,train_end_season=2025,test_season=int(target[:4])); candidates=build_candidate_frame(combined_predictions,combined_features); models,_=fit_models(candidates,ml_cfg,WORK/"moneyline_models"); scored=score_candidates(candidates,models,ml_cfg)
    cfg=MarketIntelligenceV29Config(train_start_season=2021,train_end_season=2025,test_season=int(target[:4]),snapshot_mode="live_paper"); outputs=run_v29(ml_scored_candidates=scored,ou_predictions=combined_predictions,features=combined_features,cfg=cfg)
    output=ROOT/"output"; output.mkdir(exist_ok=True)
    for name,frame in outputs.items(): frame.to_csv(output/f"{name}.csv",index=False)
    card=build_card(outputs,target); card.to_csv(output/"edge_picks.csv",index=False); write_markdown(card,target,output/"EDGE_PICKS.md")
    full=build_full_slate(outputs,target); full.to_csv(output/"all_game_predictions.csv",index=False); write_full_slate(full,target,output/"ALL_GAME_PREDICTIONS.md"); write_dashboard(full,target,output/"site")
    print(f"Finished {target}: {len(full)} games and {len(card)} edge pick(s).")

if __name__=="__main__": main()
