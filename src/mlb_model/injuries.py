from __future__ import annotations

import argparse
from io import StringIO
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from .governance import canonical_hash, utc_now_iso, write_manifest


ESPN_INJURIES_URL = "https://www.espn.com/mlb/injuries"


TEAM_ALIASES = {
    "Arizona Diamondbacks": "Arizona Diamondbacks",
    "Athletics": "Athletics",
    "Oakland Athletics": "Athletics",
    "Sacramento Athletics": "Athletics",
    "Atlanta Braves": "Atlanta Braves",
    "Baltimore Orioles": "Baltimore Orioles",
    "Boston Red Sox": "Boston Red Sox",
    "Chicago Cubs": "Chicago Cubs",
    "Chicago White Sox": "Chicago White Sox",
    "Cincinnati Reds": "Cincinnati Reds",
    "Cleveland Guardians": "Cleveland Guardians",
    "Colorado Rockies": "Colorado Rockies",
    "Detroit Tigers": "Detroit Tigers",
    "Houston Astros": "Houston Astros",
    "Kansas City Royals": "Kansas City Royals",
    "Los Angeles Angels": "Los Angeles Angels",
    "Los Angeles Dodgers": "Los Angeles Dodgers",
    "Miami Marlins": "Miami Marlins",
    "Milwaukee Brewers": "Milwaukee Brewers",
    "Minnesota Twins": "Minnesota Twins",
    "New York Mets": "New York Mets",
    "New York Yankees": "New York Yankees",
    "Philadelphia Phillies": "Philadelphia Phillies",
    "Pittsburgh Pirates": "Pittsburgh Pirates",
    "San Diego Padres": "San Diego Padres",
    "San Francisco Giants": "San Francisco Giants",
    "Seattle Mariners": "Seattle Mariners",
    "St. Louis Cardinals": "St. Louis Cardinals",
    "St Louis Cardinals": "St. Louis Cardinals",
    "Tampa Bay Rays": "Tampa Bay Rays",
    "Texas Rangers": "Texas Rangers",
    "Toronto Blue Jays": "Toronto Blue Jays",
    "Washington Nationals": "Washington Nationals",
}


POSITION_BUCKETS = {
    "SP": "sp",
    "RP": "rp",
    "P": "rp",
    "C": "bat",
    "1B": "bat",
    "2B": "bat",
    "3B": "bat",
    "SS": "bat",
    "LF": "bat",
    "CF": "bat",
    "RF": "bat",
    "OF": "bat",
    "DH": "bat",
}


def _normalize_team(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return TEAM_ALIASES.get(text, text)


def _severity(status: object) -> float:
    text = str(status or "").lower()
    if "60" in text:
        return 3.0
    if "15" in text:
        return 2.0
    if "10" in text or "out" in text or "il" in text:
        return 1.5
    if "day" in text:
        return 0.5
    return 1.0 if text else 0.0


def _team_from_heading(text: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    for team in TEAM_ALIASES:
        if cleaned == team:
            return _normalize_team(team)
    return None


def fetch_espn_injury_html(url: str = ESPN_INJURIES_URL) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 mlb-model-injuries/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_espn_injury_tables(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    team_names = [
        _team_from_heading(re.sub(r"<[^>]+>", "", match))
        for match in re.findall(r'class="injuries__teamName[^"]*"[^>]*>(.*?)</span>', html, flags=re.I | re.S)
    ]
    if not team_names:
        team_names = [_team_from_heading(match) for match in re.findall(r"<h2[^>]*>(.*?)</h2>", html, flags=re.I | re.S)]
    team_names = [team for team in team_names if team]
    rows = []
    team_idx = 0
    for table in tables:
        cols = [str(col).strip().lower() for col in table.columns]
        if not {"name", "pos", "status"}.issubset(set(cols)):
            continue
        team = team_names[team_idx] if team_idx < len(team_names) else ""
        team_idx += 1
        renamed = table.copy()
        renamed.columns = cols
        for _, item in renamed.iterrows():
            pos = str(item.get("pos", "")).upper()
            rows.append(
                {
                    "team": team,
                    "player": item.get("name"),
                    "position": pos,
                    "position_bucket": POSITION_BUCKETS.get(pos, "other"),
                    "status": item.get("status"),
                    "est_return": item.get("est. return date", item.get("est return date")),
                    "comment": item.get("comment"),
                    "injury_severity": _severity(item.get("status")),
                    "source": ESPN_INJURIES_URL,
                }
            )
    return pd.DataFrame(rows)


def summarize_injuries(injuries: pd.DataFrame, *, as_of_date: str | None = None) -> pd.DataFrame:
    if injuries.empty:
        return pd.DataFrame(columns=["team", "injury_count", "sp_injury_count", "rp_injury_count", "bat_injury_count", "injury_severity_sum"])
    out = injuries.copy()
    out["team"] = out["team"].map(_normalize_team)
    out["injury_severity"] = pd.to_numeric(out["injury_severity"], errors="coerce").fillna(0.0)
    rows = []
    for team, group in out.groupby("team", sort=True):
        row = {
            "team": team,
            "injury_count": int(len(group)),
            "sp_injury_count": int(group["position_bucket"].eq("sp").sum()),
            "rp_injury_count": int(group["position_bucket"].eq("rp").sum()),
            "bat_injury_count": int(group["position_bucket"].eq("bat").sum()),
            "injury_severity_sum": float(group["injury_severity"].sum()),
            "injury_source": ESPN_INJURIES_URL,
        }
        if as_of_date:
            row["injury_as_of_date"] = as_of_date
        rows.append(row)
    return pd.DataFrame(rows)


def attach_injury_summary(games: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    out = games.copy()
    if summary.empty:
        summary = pd.DataFrame(columns=["team"])
    right = summary.copy()
    if "team" not in right.columns:
        raise ValueError("Injury summary must include team")
    right["team_key"] = right["team"].map(_normalize_team)
    for side in ["home", "away"]:
        out[f"{side}_team_key"] = out[f"{side}_team"].map(_normalize_team)
        side_summary = right.add_prefix(f"{side}_injury_").rename(columns={f"{side}_injury_team_key": f"{side}_team_key"})
        out = out.merge(side_summary, on=f"{side}_team_key", how="left")
        out = out.drop(columns=[f"{side}_team_key"], errors="ignore")
    numeric = [
        "injury_count",
        "sp_injury_count",
        "rp_injury_count",
        "bat_injury_count",
        "injury_severity_sum",
    ]
    for side in ["home", "away"]:
        for col in numeric:
            full = f"{side}_injury_{col}"
            if full not in out.columns:
                out[full] = 0.0
            out[full] = pd.to_numeric(out[full], errors="coerce").fillna(0.0)
    out["injury_count_diff"] = out["home_injury_injury_count"] - out["away_injury_injury_count"]
    out["sp_injury_count_diff"] = out["home_injury_sp_injury_count"] - out["away_injury_sp_injury_count"]
    out["rp_injury_count_diff"] = out["home_injury_rp_injury_count"] - out["away_injury_rp_injury_count"]
    out["bat_injury_count_diff"] = out["home_injury_bat_injury_count"] - out["away_injury_bat_injury_count"]
    out["injury_severity_diff"] = out["home_injury_injury_severity_sum"] - out["away_injury_injury_severity_sum"]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch/summarize ESPN MLB injuries and optionally join them to game rows.")
    parser.add_argument("--html", type=Path, help="Existing ESPN injuries HTML file.")
    parser.add_argument("--injuries", type=Path, help="Existing parsed injury CSV.")
    parser.add_argument("--games", type=Path, help="Optional game CSV to enrich.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--raw-html-out", type=Path)
    parser.add_argument("--as-of-date")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.injuries:
        injuries = pd.read_csv(args.injuries)
    else:
        html = args.html.read_text(encoding="utf-8") if args.html else fetch_espn_injury_html()
        if args.raw_html_out:
            args.raw_html_out.parent.mkdir(parents=True, exist_ok=True)
            args.raw_html_out.write_text(html, encoding="utf-8")
        injuries = parse_espn_injury_tables(html)

    summary = summarize_injuries(injuries, as_of_date=args.as_of_date)
    output = attach_injury_summary(pd.read_csv(args.games), summary) if args.games else summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    write_manifest(
        args.out.with_suffix(".manifest.json"),
        {
            "generated_at_utc": utc_now_iso(),
            "source": ESPN_INJURIES_URL,
            "injury_rows": len(injuries),
            "rows": len(output),
            "joined_to_games": bool(args.games),
            "output_hash": canonical_hash(output),
        },
    )
    print(f"Wrote {len(output)} rows to {args.out}")


if __name__ == "__main__":
    main()
