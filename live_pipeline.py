from __future__ import annotations

import json, os, re, sys
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
    print(f"Finished {target}: {len(card)} edge pick(s).")

if __name__=="__main__": main()
