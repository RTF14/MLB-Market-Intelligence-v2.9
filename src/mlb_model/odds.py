from __future__ import annotations

import argparse
from datetime import timedelta
import os
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


THE_ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
THE_ODDS_API_HISTORICAL_BASE_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"
ODDSCHECKER_MLB_URL = "https://www.oddschecker.com/us/baseball/mlb"
API_SPORTS_BASEBALL_BASE_URL = "https://v1.baseball.api-sports.io"


TEAM_ALIASES = {
    "ARI": "Arizona Diamondbacks",
    "AZ": "Arizona Diamondbacks",
    "Arizona": "Arizona Diamondbacks",
    "Arizona D'Backs": "Arizona Diamondbacks",
    "Arizona Diamondbacks": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "ATL ": "Atlanta Braves",
    "Atlanta": "Atlanta Braves",
    "Atlanta Braves": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BAL ": "Baltimore Orioles",
    "Baltimore": "Baltimore Orioles",
    "Baltimore Orioles": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "BOS ": "Boston Red Sox",
    "Boston": "Boston Red Sox",
    "Boston Red Sox": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CHC ": "Chicago Cubs",
    "Chi Cubs": "Chicago Cubs",
    "Chicago Cubs": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "CHW": "Chicago White Sox",
    "CHW ": "Chicago White Sox",
    "Chi Sox": "Chicago White Sox",
    "Chicago White Sox": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CIN ": "Cincinnati Reds",
    "CIN -": "Cincinnati Reds",
    "Cincinnati": "Cincinnati Reds",
    "Cincinnati Reds": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "Cleveland": "Cleveland Guardians",
    "Cleveland Indians": "Cleveland Guardians",
    "Cleveland Guardians": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "COL ": "Colorado Rockies",
    "Colorado": "Colorado Rockies",
    "Colorado Rockies": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "DET ": "Detroit Tigers",
    "Detroit": "Detroit Tigers",
    "Detroit Tigers": "Detroit Tigers",
    "HOU": "Houston Astros",
    "HOU ": "Houston Astros",
    "Houston": "Houston Astros",
    "Houston Astros": "Houston Astros",
    "KC": "Kansas City Royals",
    "KC ": "Kansas City Royals",
    "KCR": "Kansas City Royals",
    "Kansas City": "Kansas City Royals",
    "Kansas City Royals": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAA ": "Los Angeles Angels",
    "LA Angels": "Los Angeles Angels",
    "Los Angeles Angels": "Los Angeles Angels",
    "ANA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "LAD ": "Los Angeles Dodgers",
    "LA Dodgers": "Los Angeles Dodgers",
    "Los Angeles Dodgers": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIA ": "Miami Marlins",
    "Miami": "Miami Marlins",
    "Florida Marlins": "Miami Marlins",
    "FLO": "Miami Marlins",
    "Miami Marlins": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIL ": "Milwaukee Brewers",
    "Milwaukee": "Milwaukee Brewers",
    "Milwaukee Brewers": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "MIN ": "Minnesota Twins",
    "Minnesota": "Minnesota Twins",
    "Minnesota Twins": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYM ": "New York Mets",
    "NY Mets": "New York Mets",
    "New York Mets": "New York Mets",
    "NYY": "New York Yankees",
    "NYY ": "New York Yankees",
    "NY Yankees": "New York Yankees",
    "New York Yankees": "New York Yankees",
    "OAK": "Athletics",
    "ATH": "Athletics",
    "Oakland": "Athletics",
    "Oakland Athletics": "Athletics",
    "A's": "Athletics",
    "Athletics": "Athletics",
    "PHI": "Philadelphia Phillies",
    "PHI ": "Philadelphia Phillies",
    "Philadelphia": "Philadelphia Phillies",
    "Philadelphia Phillies": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "PIT ": "Pittsburgh Pirates",
    "Pittsburgh": "Pittsburgh Pirates",
    "Pittsburgh Pirates": "Pittsburgh Pirates",
    "SD": "San Diego Padres",
    "SD ": "San Diego Padres",
    "SDP": "San Diego Padres",
    "San Diego": "San Diego Padres",
    "San Diego Padres": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SEA ": "Seattle Mariners",
    "Seattle": "Seattle Mariners",
    "Seattle Mariners": "Seattle Mariners",
    "SF": "San Francisco Giants",
    "SF ": "San Francisco Giants",
    "SFG": "San Francisco Giants",
    "San Francisco": "San Francisco Giants",
    "San Francisco Giants": "San Francisco Giants",
    "STL": "St. Louis Cardinals",
    "STL ": "St. Louis Cardinals",
    "St Louis": "St. Louis Cardinals",
    "St. Louis": "St. Louis Cardinals",
    "St. Louis Cardinals": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays",
    "TB ": "Tampa Bay Rays",
    "TBR": "Tampa Bay Rays",
    "Tampa Bay": "Tampa Bay Rays",
    "Tampa Bay Rays": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TEX ": "Texas Rangers",
    "Texas": "Texas Rangers",
    "Texas Rangers": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "TOR ": "Toronto Blue Jays",
    "Toronto": "Toronto Blue Jays",
    "Toronto Blue Jays": "Toronto Blue Jays",
    "WAS": "Washington Nationals",
    "WAS ": "Washington Nationals",
    "WSH": "Washington Nationals",
    "Washington": "Washington Nationals",
    "Washington Nationals": "Washington Nationals",
}


def canonical_team(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = " ".join(text.replace("@", "").split())
    return TEAM_ALIASES.get(text, text)


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {column.lower().strip(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def normalize_sports_odds_history(odds: pd.DataFrame) -> pd.DataFrame:
    out = odds.copy()
    date_col = _first_existing(out, ["game_date", "date", "Date", "Game Date"])
    away_col = _first_existing(out, ["away_team", "away", "visitor", "Visitor", "Away Team", "Team"])
    home_col = _first_existing(out, ["home_team", "home", "Home Team", "Opponent"])
    total_col = _first_existing(out, ["total_line", "closing_total", "close_total", "Close Total", "Total", "O/U", "OU"])
    over_price_col = _first_existing(out, ["total_price_over", "over_price", "Over Price", "O$"])
    under_price_col = _first_existing(out, ["total_price_under", "under_price", "Under Price", "U$"])

    missing = []
    if date_col is None:
        missing.append("date")
    if away_col is None:
        missing.append("away_team")
    if home_col is None:
        missing.append("home_team")
    if total_col is None:
        missing.append("total_line")
    if missing:
        raise ValueError(f"SportsOddsHistory file missing recognizable columns: {missing}")

    normalized = pd.DataFrame(
        {
            "game_date": pd.to_datetime(out[date_col], errors="raise").dt.date.astype(str),
            "away_team": out[away_col].map(canonical_team),
            "home_team": out[home_col].map(canonical_team),
            "total_line": pd.to_numeric(out[total_col], errors="coerce"),
            "sportsbook": "SportsOddsHistory",
        }
    )
    if over_price_col:
        normalized["total_price_over"] = pd.to_numeric(out[over_price_col], errors="coerce")
    if under_price_col:
        normalized["total_price_under"] = pd.to_numeric(out[under_price_col], errors="coerce")
    normalized = normalized.dropna(subset=["total_line"])
    return normalized.drop_duplicates(["game_date", "away_team", "home_team"], keep="last")


def normalize_oddsportal(odds: pd.DataFrame) -> pd.DataFrame:
    out = odds.copy()
    date_col = _first_existing(out, ["game_date", "date", "Date", "Match Date"])
    away_col = _first_existing(out, ["away_team", "away", "visitor", "Visitor", "Away", "Away Team"])
    home_col = _first_existing(out, ["home_team", "home", "Home", "Home Team"])
    matchup_col = _first_existing(out, ["match", "Match", "event", "Event", "teams", "Teams"])
    total_col = _first_existing(
        out,
        [
            "total_line",
            "closing_total",
            "close_total",
            "Close Total",
            "Total",
            "O/U",
            "Over/Under",
            "over_under",
        ],
    )
    over_price_col = _first_existing(out, ["total_price_over", "over_price", "Over", "Over Odds"])
    under_price_col = _first_existing(out, ["total_price_under", "under_price", "Under", "Under Odds"])
    home_ml_col = _first_existing(out, ["home_moneyline", "home_ml", "Home ML", "1"])
    away_ml_col = _first_existing(out, ["away_moneyline", "away_ml", "Away ML", "2"])

    missing = []
    if date_col is None:
        missing.append("date")
    if away_col is None and matchup_col is None:
        missing.append("away_team")
    if home_col is None and matchup_col is None:
        missing.append("home_team")
    if total_col is None:
        missing.append("total_line")
    if missing:
        raise ValueError(f"OddsPortal file missing recognizable columns: {missing}")

    if away_col and home_col:
        away = out[away_col].map(canonical_team)
        home = out[home_col].map(canonical_team)
    else:
        split = out[matchup_col].astype(str).str.split(r"\s+[-–]\s+|\s+@\s+", n=1, regex=True, expand=True)
        if split.shape[1] < 2:
            raise ValueError("Could not split OddsPortal matchup column into two teams")
        away = split[0].map(canonical_team)
        home = split[1].map(canonical_team)

    normalized = pd.DataFrame(
        {
            "game_date": pd.to_datetime(out[date_col], errors="raise").dt.date.astype(str),
            "away_team": away,
            "home_team": home,
            "total_line": pd.to_numeric(out[total_col], errors="coerce"),
            "sportsbook": "OddsPortal",
        }
    )
    if over_price_col:
        normalized["total_price_over"] = pd.to_numeric(out[over_price_col], errors="coerce")
    if under_price_col:
        normalized["total_price_under"] = pd.to_numeric(out[under_price_col], errors="coerce")
    if home_ml_col:
        normalized["home_moneyline"] = pd.to_numeric(out[home_ml_col], errors="coerce")
    if away_ml_col:
        normalized["away_moneyline"] = pd.to_numeric(out[away_ml_col], errors="coerce")
    normalized = normalized.dropna(subset=["total_line"])
    return normalized.drop_duplicates(["game_date", "away_team", "home_team"], keep="last")


def _parse_oddschecker_american(value: str) -> float | None:
    text = str(value).strip().replace("−", "-")
    match = re.search(r"([+-]\d+)$", text)
    if not match:
        return None
    return float(match.group(1))


def _fractional_to_american(value: str) -> float | None:
    text = str(value).strip()
    match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", text)
    if not match:
        return None
    numerator = float(match.group(1))
    denominator = float(match.group(2))
    if denominator <= 0:
        return None
    decimal_profit = numerator / denominator
    if decimal_profit >= 1:
        return round(decimal_profit * 100.0, 3)
    return round(-100.0 / decimal_profit, 3)


def _parse_oddschecker_total(value: str) -> tuple[float | None, float | None]:
    text = str(value).strip().replace("−", "-")
    match = re.fullmatch(r"[OU]\s*([0-9]+(?:\.[0-9]+)?)([+-]\d+)", text, flags=re.IGNORECASE)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def _parse_oddschecker_spread(value: str) -> tuple[float | None, float | None]:
    text = str(value).strip().replace("−", "-")
    match = re.fullmatch(r"([+-]?[0-9]+(?:\.[0-9]+)?)([+-]\d+)", text)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def _split_oddschecker_matchup(value: str) -> tuple[str, str] | None:
    text = " ".join(str(value).strip().split())
    if " @ " in text:
        away, home = [canonical_team(part) for part in text.split(" @ ", 1)]
        return away, home
    team_names = sorted(set(TEAM_ALIASES.values()), key=len, reverse=True)
    for away in team_names:
        prefix = f"{away} "
        if text.startswith(prefix):
            home = canonical_team(text[len(prefix) :])
            if home in team_names:
                return away, home
    return None


def normalize_oddschecker_html(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in re.sub(r"<[^>]+>", "\n", text).splitlines()]
    lines = [line for line in lines if line]
    rows: list[dict[str, object]] = []
    current_date = ""
    i = 0
    while i < len(lines):
        date_match = re.fullmatch(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(.+)", lines[i])
        compact_date_match = re.fullmatch(
            r"MLB\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+(.+)",
            lines[i],
            flags=re.IGNORECASE,
        )
        if date_match:
            current_date = lines[i]
            i += 1
            continue
        if compact_date_match:
            current_date = lines[i]
            i += 1
            continue
        matchup = _split_oddschecker_matchup(lines[i])
        if matchup is None:
            i += 1
            continue
        away_team, home_team = matchup
        if not away_team or not home_team:
            i += 1
            continue
        ml_away = _parse_oddschecker_american(lines[i + 1]) or _fractional_to_american(lines[i + 1])
        ml_home = _parse_oddschecker_american(lines[i + 2]) or _fractional_to_american(lines[i + 2])
        away_spread, away_spread_price = (None, None)
        home_spread, home_spread_price = (None, None)
        over_total, over_price = (None, None)
        under_total, under_price = (None, None)
        if i + 6 < len(lines):
            away_spread, away_spread_price = _parse_oddschecker_spread(lines[i + 3])
            home_spread, home_spread_price = _parse_oddschecker_spread(lines[i + 4])
            over_total, over_price = _parse_oddschecker_total(lines[i + 5])
            under_total, under_price = _parse_oddschecker_total(lines[i + 6])
        if ml_away is None and ml_home is None and over_total is None and under_total is None:
            i += 1
            continue
        has_full_us_market = any(value is not None for value in [away_spread, home_spread, over_total, under_total])
        rows.append(
            {
                "game_date_label": current_date,
                "away_team": away_team,
                "home_team": home_team,
                "sportsbook": "OddsCheckerCurrent",
                "total_line": over_total if over_total is not None else under_total,
                "total_price_over": over_price,
                "total_price_under": under_price,
                "home_moneyline": ml_home,
                "away_moneyline": ml_away,
                "home_spread": home_spread,
                "away_spread": away_spread,
                "home_spread_price": home_spread_price,
                "away_spread_price": away_spread_price,
            }
        )
        i += 7 if has_full_us_market else 3
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.drop_duplicates(["away_team", "home_team"], keep="last")


def _default_browser_path() -> str | None:
    candidates = [
        shutil.which("chrome"),
        shutil.which("chrome.exe"),
        shutil.which("msedge"),
        shutil.which("msedge.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def fetch_oddschecker_html_with_browser(
    out_path: Path,
    *,
    url: str = ODDSCHECKER_MLB_URL,
    browser_path: str | None = None,
    timeout_seconds: int = 60,
) -> None:
    resolved_browser = browser_path or _default_browser_path()
    if not resolved_browser:
        raise ValueError("Could not find Chrome or Edge. Pass --oddschecker-browser-path.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlb_oddschecker_browser_") as user_data_dir:
        cmd = [
            resolved_browser,
            "--headless=new",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={user_data_dir}",
            "--window-size=1440,2200",
            "--virtual-time-budget=10000",
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "--dump-dom",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Browser-backed OddsChecker fetch failed with exit {result.returncode}: {detail[:1000]}")
    payload = result.stdout.strip()
    if not payload or "MLB" not in payload:
        raise RuntimeError("Browser-backed OddsChecker fetch returned an empty or unexpected page")
    out_path.write_text(payload, encoding="utf-8")


def _parse_yyyymmdd(value: pd.Series) -> pd.Series:
    text = value.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    eight_digit = text.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=value.index, dtype="datetime64[ns]")
    if eight_digit.any():
        parsed.loc[eight_digit] = pd.to_datetime(text.loc[eight_digit], format="%Y%m%d", errors="coerce")
    if (~eight_digit).any():
        parsed.loc[~eight_digit] = pd.to_datetime(text.loc[~eight_digit], errors="coerce")
    if parsed.isna().any():
        examples = text.loc[parsed.isna()].head(5).tolist()
        raise ValueError(f"Historical odds file has invalid date values: {examples}")
    return parsed


def normalize_historical_basic(odds: pd.DataFrame) -> pd.DataFrame:
    out = odds.copy()
    required = [
        "game id",
        "date",
        "away team",
        "away score",
        "away ml open",
        "away ml close",
        "over open",
        "over open odds",
        "over close",
        "over close odds",
        "home team",
        "home score",
        "home ml open",
        "home ml close",
        "under open",
        "under open odds",
        "under close",
        "under close odds",
    ]
    lowered = {column.lower().strip(): column for column in out.columns}
    missing = [name for name in required if name not in lowered]
    if missing:
        raise ValueError(f"Historical basic odds file missing columns: {missing}")

    def col(name: str) -> str:
        return lowered[name]

    normalized = pd.DataFrame(
        {
            "historical_odds_game_id": out[col("game id")],
            "game_date": _parse_yyyymmdd(out[col("date")]).dt.date.astype(str),
            "away_team": out[col("away team")].map(canonical_team),
            "home_team": out[col("home team")].map(canonical_team),
            "away_score_odds": pd.to_numeric(out[col("away score")], errors="coerce"),
            "home_score_odds": pd.to_numeric(out[col("home score")], errors="coerce"),
            "away_moneyline_open": pd.to_numeric(out[col("away ml open")], errors="coerce"),
            "away_moneyline": pd.to_numeric(out[col("away ml close")], errors="coerce"),
            "home_moneyline_open": pd.to_numeric(out[col("home ml open")], errors="coerce"),
            "home_moneyline": pd.to_numeric(out[col("home ml close")], errors="coerce"),
            "total_line_open": pd.to_numeric(out[col("over open")], errors="coerce"),
            "total_price_over_open": pd.to_numeric(out[col("over open odds")], errors="coerce"),
            "total_price_under_open": pd.to_numeric(out[col("under open odds")], errors="coerce"),
            "total_line": pd.to_numeric(out[col("over close")], errors="coerce"),
            "total_line_under_close": pd.to_numeric(out[col("under close")], errors="coerce"),
            "total_price_over": pd.to_numeric(out[col("over close odds")], errors="coerce"),
            "total_price_under": pd.to_numeric(out[col("under close odds")], errors="coerce"),
            "sportsbook": "HistoricalBasic",
        }
    )
    mismatch = (
        normalized["total_line"].notna()
        & normalized["total_line_under_close"].notna()
        & (normalized["total_line"] != normalized["total_line_under_close"])
    )
    if mismatch.any():
        examples = normalized.loc[mismatch, ["game_date", "away_team", "home_team", "total_line", "total_line_under_close"]].head(5)
        raise ValueError(f"Historical basic over/under close totals disagree: {examples.to_dict('records')}")
    normalized["total_line_move"] = normalized["total_line"] - normalized["total_line_open"]
    normalized["total_line_source"] = "close"
    normalized = normalized.drop(columns=["total_line_under_close"])
    normalized = normalized.dropna(subset=["game_date", "away_team", "home_team", "total_line"])
    return normalized.drop_duplicates(["game_date", "away_team", "home_team"], keep="last")


def _current_or_opening(line: dict, key: str) -> object:
    current = line.get("currentLine") or {}
    opening = line.get("openingLine") or {}
    return current.get(key, opening.get(key))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return None
    return float(series.median())


def normalize_sbr_json(payload: dict, *, sportsbook: str = "consensus") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sportsbook = sportsbook.lower()
    for game_date, games in payload.items():
        for game in games or []:
            view = game.get("gameView") or {}
            odds = game.get("odds") or {}
            away_team = canonical_team((view.get("awayTeam") or {}).get("fullName"))
            home_team = canonical_team((view.get("homeTeam") or {}).get("fullName"))
            if not away_team or not home_team:
                continue

            totals = odds.get("totals") or []
            moneylines = odds.get("moneyline") or []
            spreads = odds.get("pointspread") or []
            if sportsbook != "consensus":
                totals = [line for line in totals if str(line.get("sportsbook", "")).lower() == sportsbook]
                moneylines = [line for line in moneylines if str(line.get("sportsbook", "")).lower() == sportsbook]
                spreads = [line for line in spreads if str(line.get("sportsbook", "")).lower() == sportsbook]

            total_values = [
                float(value)
                for value in (_current_or_opening(line, "total") for line in totals)
                if value not in [None, ""]
            ]
            over_prices = [
                float(value)
                for value in (_current_or_opening(line, "overOdds") for line in totals)
                if value not in [None, ""]
            ]
            under_prices = [
                float(value)
                for value in (_current_or_opening(line, "underOdds") for line in totals)
                if value not in [None, ""]
            ]
            home_ml = [
                float(value)
                for value in (_current_or_opening(line, "homeOdds") for line in moneylines)
                if value not in [None, ""]
            ]
            away_ml = [
                float(value)
                for value in (_current_or_opening(line, "awayOdds") for line in moneylines)
                if value not in [None, ""]
            ]
            home_spread = [
                float(value)
                for value in (_current_or_opening(line, "homeSpread") for line in spreads)
                if value not in [None, ""]
            ]
            away_spread = [
                float(value)
                for value in (_current_or_opening(line, "awaySpread") for line in spreads)
                if value not in [None, ""]
            ]
            home_spread_price = [
                float(value)
                for value in (_current_or_opening(line, "homeOdds") for line in spreads)
                if value not in [None, ""]
            ]
            away_spread_price = [
                float(value)
                for value in (_current_or_opening(line, "awayOdds") for line in spreads)
                if value not in [None, ""]
            ]

            rows.append(
                {
                    "game_date": pd.to_datetime(game_date, errors="raise").date().isoformat(),
                    "away_team": away_team,
                    "home_team": home_team,
                    "away_score": view.get("awayTeamScore"),
                    "home_score": view.get("homeTeamScore"),
                    "game_type": view.get("gameType"),
                    "sportsbook": sportsbook,
                    "total_line": _median(total_values),
                    "total_price_over": _median(over_prices),
                    "total_price_under": _median(under_prices),
                    "home_moneyline": _median(home_ml),
                    "away_moneyline": _median(away_ml),
                    "home_spread": _median(home_spread),
                    "away_spread": _median(away_spread),
                    "home_spread_price": _median(home_spread_price),
                    "away_spread_price": _median(away_spread_price),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.drop_duplicates(["game_date", "away_team", "home_team"], keep="last")


def _market_by_key(bookmaker: dict) -> dict[str, dict]:
    return {str(market.get("key")): market for market in bookmaker.get("markets") or []}


def _outcome_price(market: dict | None, outcome_name: str) -> float | None:
    if not market:
        return None
    target = canonical_team(outcome_name)
    for outcome in market.get("outcomes") or []:
        name = canonical_team(outcome.get("name"))
        if name == target and outcome.get("price") not in [None, ""]:
            return float(outcome["price"])
    return None


def _totals_value(market: dict | None, outcome_name: str, key: str) -> float | None:
    if not market:
        return None
    for outcome in market.get("outcomes") or []:
        if str(outcome.get("name", "")).lower() == outcome_name and outcome.get(key) not in [None, ""]:
            return float(outcome[key])
    return None


def _the_odds_api_game_date(commence_time: object) -> str:
    parsed = pd.to_datetime(commence_time, errors="raise", utc=True)
    return (parsed - timedelta(hours=12)).date().isoformat()


def _the_odds_api_book_rows(event: dict) -> list[dict[str, object]]:
    away_team = canonical_team(event.get("away_team"))
    home_team = canonical_team(event.get("home_team"))
    if not away_team or not home_team:
        return []
    commence_time = event.get("commence_time")
    game_date = _the_odds_api_game_date(commence_time)
    rows = []
    for bookmaker in event.get("bookmakers") or []:
        markets = _market_by_key(bookmaker)
        totals = markets.get("totals")
        total_line = _totals_value(totals, "over", "point")
        under_total_line = _totals_value(totals, "under", "point")
        if total_line is None:
            total_line = under_total_line
        rows.append(
            {
                "source_event_id": event.get("id"),
                "game_date": game_date,
                "commence_time_utc": pd.to_datetime(commence_time, errors="raise", utc=True).isoformat(),
                "away_team": away_team,
                "home_team": home_team,
                "sportsbook": str(bookmaker.get("key") or bookmaker.get("title") or "").lower(),
                "sportsbook_title": bookmaker.get("title"),
                "market_timestamp_utc": bookmaker.get("last_update"),
                "total_line": total_line,
                "total_price_over": _totals_value(totals, "over", "price"),
                "total_price_under": _totals_value(totals, "under", "price"),
                "home_moneyline": _outcome_price(markets.get("h2h"), home_team),
                "away_moneyline": _outcome_price(markets.get("h2h"), away_team),
            }
        )
    return rows


def normalize_the_odds_api_json(payload: list[dict] | dict, *, sportsbook: str = "consensus") -> pd.DataFrame:
    if isinstance(payload, dict):
        payload = payload.get("data", [])
    book_rows: list[dict[str, object]] = []
    for event in payload or []:
        book_rows.extend(_the_odds_api_book_rows(event))
    books = pd.DataFrame(book_rows)
    if books.empty:
        return books

    sportsbook = sportsbook.lower()
    if sportsbook == "all":
        return books.drop_duplicates(["source_event_id", "sportsbook"], keep="last")

    if sportsbook != "consensus":
        title = books["sportsbook_title"].astype(str).str.lower()
        key = books["sportsbook"].astype(str).str.lower()
        books = books[key.eq(sportsbook) | title.eq(sportsbook)]
        return books.drop_duplicates(["source_event_id", "sportsbook"], keep="last")

    rows = []
    for _, group in books.groupby(["source_event_id", "game_date", "away_team", "home_team"], sort=False):
        rows.append(
            {
                "source_event_id": group["source_event_id"].iloc[0],
                "game_date": group["game_date"].iloc[0],
                "commence_time_utc": group["commence_time_utc"].iloc[0],
                "away_team": group["away_team"].iloc[0],
                "home_team": group["home_team"].iloc[0],
                "sportsbook": "the-odds-api-consensus",
                "sportsbook_count": int(group["sportsbook"].nunique()),
                "market_timestamp_utc": group["market_timestamp_utc"].dropna().max(),
                "total_line": _median(group["total_line"].dropna().astype(float).tolist()),
                "total_price_over": _median(group["total_price_over"].dropna().astype(float).tolist()),
                "total_price_under": _median(group["total_price_under"].dropna().astype(float).tolist()),
                "home_moneyline": _median(group["home_moneyline"].dropna().astype(float).tolist()),
                "away_moneyline": _median(group["away_moneyline"].dropna().astype(float).tolist()),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["game_date", "away_team", "home_team"], keep="last")


def fetch_the_odds_api_json(
    api_key: str,
    *,
    regions: str = "us",
    markets: str = "h2h,totals",
    odds_format: str = "american",
    bookmakers: str | None = None,
    snapshot_date: str | None = None,
    timeout_seconds: int = 30,
) -> list[dict] | dict:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    base_url = THE_ODDS_API_BASE_URL
    if snapshot_date:
        params["date"] = snapshot_date
        base_url = THE_ODDS_API_HISTORICAL_BASE_URL
    url = f"{base_url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "mlb-model/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"The Odds API request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"The Odds API request failed: {exc.reason}") from exc


def _decimal_to_american(value: object) -> float | None:
    try:
        decimal = float(value)
    except (TypeError, ValueError):
        return None
    if decimal <= 1:
        return None
    profit = decimal - 1.0
    if profit >= 1.0:
        return round(profit * 100.0, 3)
    return round(-100.0 / profit, 3)


def _api_sports_team_name(game: dict, side: str) -> str:
    teams = game.get("teams") or {}
    team = teams.get(side) or game.get(side) or {}
    if isinstance(team, dict):
        return canonical_team(team.get("name"))
    return canonical_team(team)


def _api_sports_game_date(game: dict) -> str:
    date_value = game.get("date") or game.get("datetime") or game.get("time")
    if date_value:
        return pd.to_datetime(date_value, errors="raise", utc=True).date().isoformat()
    return ""


def _parse_api_sports_total_label(value: object) -> tuple[str | None, float | None]:
    text = str(value or "").strip()
    match = re.search(r"\b(over|under)\b\s*([0-9]+(?:\.[0-9]+)?)?", text, flags=re.IGNORECASE)
    if not match:
        return None, None
    side = match.group(1).upper()
    point = float(match.group(2)) if match.group(2) else None
    return side, point


def _api_sports_bet_kind(bet: dict) -> str:
    bet_id = str(bet.get("id") or "").strip()
    text = str(bet.get("name") or "").strip().lower()
    if bet_id == "5" or text == "over/under":
        return "TOTAL"
    if bet_id == "1" or text in {"home/away", "moneyline"}:
        return "MONEYLINE"
    if bet_id == "2" or text in {"asian handicap", "spread", "run line"}:
        return "SPREAD"
    return ""


def normalize_api_sports_baseball_json(payload: dict, *, sportsbook: str = "consensus") -> pd.DataFrame:
    responses = payload.get("response", payload if isinstance(payload, list) else [])
    book_rows: list[dict[str, object]] = []
    for item in responses or []:
        game = item.get("game") or item.get("fixture") or item.get("match") or {}
        away_team = _api_sports_team_name(game, "away")
        home_team = _api_sports_team_name(game, "home")
        if not away_team or not home_team:
            continue
        base = {
            "source_event_id": game.get("id") or game.get("game") or item.get("id"),
            "game_date": _api_sports_game_date(game),
            "away_team": away_team,
            "home_team": home_team,
        }
        for bookmaker in item.get("bookmakers") or []:
            row = {
                **base,
                "sportsbook": str(bookmaker.get("name") or bookmaker.get("id") or "").lower(),
                "sportsbook_title": bookmaker.get("name"),
            }
            for bet in bookmaker.get("bets") or []:
                kind = _api_sports_bet_kind(bet)
                for value in bet.get("values") or []:
                    label = str(value.get("value") or value.get("name") or "").strip()
                    american = _decimal_to_american(value.get("odd") or value.get("odds"))
                    if american is None:
                        continue
                    lower_label = label.lower()
                    if kind == "MONEYLINE":
                        if lower_label in {"home", "1"} or canonical_team(label) == home_team:
                            row["home_moneyline"] = american
                        elif lower_label in {"away", "2"} or canonical_team(label) == away_team:
                            row["away_moneyline"] = american
                    elif kind == "TOTAL":
                        total_side, point = _parse_api_sports_total_label(label)
                        if point is not None:
                            row["total_line"] = point
                        if total_side == "OVER":
                            row["total_price_over"] = american
                        elif total_side == "UNDER":
                            row["total_price_under"] = american
                    elif kind == "SPREAD":
                        if canonical_team(label) == home_team or lower_label.startswith("home"):
                            row["home_spread_price"] = american
                        elif canonical_team(label) == away_team or lower_label.startswith("away"):
                            row["away_spread_price"] = american
            book_rows.append(row)
    books = pd.DataFrame(book_rows)
    if books.empty:
        return pd.DataFrame(
            columns=[
                "source_event_id",
                "game_date",
                "away_team",
                "home_team",
                "sportsbook",
                "sportsbook_count",
                "total_line",
                "total_price_over",
                "total_price_under",
                "home_moneyline",
                "away_moneyline",
            ]
        )

    sportsbook = sportsbook.lower()
    if sportsbook != "consensus":
        key = books["sportsbook"].astype(str).str.lower()
        title = books["sportsbook_title"].astype(str).str.lower()
        books = books[key.eq(sportsbook) | title.eq(sportsbook)]
        return books.drop_duplicates(["game_date", "away_team", "home_team", "sportsbook"], keep="last")

    rows = []
    for _, group in books.groupby(["game_date", "away_team", "home_team"], sort=False):
        rows.append(
            {
                "source_event_id": group["source_event_id"].dropna().iloc[0] if group["source_event_id"].notna().any() else None,
                "game_date": group["game_date"].iloc[0],
                "away_team": group["away_team"].iloc[0],
                "home_team": group["home_team"].iloc[0],
                "sportsbook": "api-sports-baseball-consensus",
                "sportsbook_count": int(group["sportsbook"].nunique()),
                "total_line": _median(group["total_line"].dropna().astype(float).tolist()) if "total_line" in group else None,
                "total_price_over": _median(group["total_price_over"].dropna().astype(float).tolist()) if "total_price_over" in group else None,
                "total_price_under": _median(group["total_price_under"].dropna().astype(float).tolist()) if "total_price_under" in group else None,
                "home_moneyline": _median(group["home_moneyline"].dropna().astype(float).tolist()) if "home_moneyline" in group else None,
                "away_moneyline": _median(group["away_moneyline"].dropna().astype(float).tolist()) if "away_moneyline" in group else None,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["game_date", "away_team", "home_team"], keep="last")


def fetch_api_sports_baseball_json(
    api_key: str,
    *,
    endpoint: str = "odds",
    base_url: str = API_SPORTS_BASEBALL_BASE_URL,
    date: str | None = None,
    game: str | None = None,
    league: str | None = None,
    season: str | None = None,
    bookmaker: str | None = None,
    bet: str | None = None,
    timeout_seconds: int = 30,
) -> dict:
    params = {}
    for key, value in {
        "date": date,
        "game": game,
        "league": league,
        "season": season,
        "bookmaker": bookmaker,
        "bet": bet,
    }.items():
        if value not in [None, ""]:
            params[key] = value
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"x-apisports-key": api_key})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API-Sports Baseball request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"API-Sports Baseball request failed: {exc.reason}") from exc


def attach_game_ids(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    game_frame = games.copy()
    game_frame["game_date"] = pd.to_datetime(game_frame["game_date"], errors="raise").dt.date.astype(str)
    game_frame["away_team"] = game_frame["away_team"].map(canonical_team)
    game_frame["home_team"] = game_frame["home_team"].map(canonical_team)
    game_cols = ["game_pk", "game_date", "away_team", "home_team"]
    if "season" in game_frame.columns:
        game_cols.insert(1, "season")
    merged = odds.merge(game_frame[game_cols], on=["game_date", "away_team", "home_team"], how="left")
    return merged


def normalize_totals(odds: pd.DataFrame) -> pd.DataFrame:
    out = odds.copy()
    game_key = "game_pk" if "game_pk" in out.columns else None
    if game_key is None:
        required = {"game_date", "home_team", "away_team"}
        missing = required - set(out.columns)
        if missing:
            raise ValueError(f"Odds file needs game_pk or date/team keys. Missing: {sorted(missing)}")
    if "total_line" not in out.columns:
        raise ValueError("Odds file must include total_line")
    out["total_line"] = pd.to_numeric(out["total_line"], errors="raise")
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="raise").dt.date.astype(str)
    keep = [
        col
        for col in [
            "game_pk",
            "game_date",
            "home_team",
            "away_team",
            "sportsbook",
            "total_line",
            "total_line_open",
            "total_line_move",
            "total_line_source",
            "total_price_over",
            "total_price_under",
            "total_price_over_open",
            "total_price_under_open",
            "home_moneyline",
            "away_moneyline",
            "home_moneyline_open",
            "away_moneyline_open",
            "home_spread",
            "away_spread",
            "home_spread_price",
            "away_spread_price",
            "market_timestamp_utc",
            "commence_time_utc",
            "source_event_id",
            "sportsbook_count",
            "historical_odds_game_id",
            "home_score_odds",
            "away_score_odds",
        ]
        if col in out.columns
    ]
    return out[keep].drop_duplicates()


def attach_totals(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    totals = normalize_totals(odds)
    preds = predictions.copy()
    if "game_date" in preds.columns:
        preds["game_date"] = pd.to_datetime(preds["game_date"], errors="raise").dt.date.astype(str)
    if "game_pk" in totals.columns and "game_pk" in preds.columns:
        return preds.merge(totals, on="game_pk", how="left", suffixes=("", "_odds"))
    keys = ["game_date", "home_team", "away_team"]
    return preds.merge(totals, on=keys, how="left", suffixes=("", "_odds"))


def load_odds(path: Path, *, fmt: str, sportsbook: str = "consensus") -> pd.DataFrame:
    if fmt == "sbr-json":
        return normalize_sbr_json(json.loads(path.read_text(encoding="utf-8")), sportsbook=sportsbook)
    if fmt == "the-odds-api-json":
        return normalize_the_odds_api_json(json.loads(path.read_text(encoding="utf-8")), sportsbook=sportsbook)
    if fmt == "api-sports-baseball-json":
        return normalize_api_sports_baseball_json(json.loads(path.read_text(encoding="utf-8")), sportsbook=sportsbook)
    if fmt == "oddschecker-html":
        return normalize_oddschecker_html(path)
    odds = pd.read_csv(path)
    if fmt == "sports-odds-history":
        return normalize_sports_odds_history(odds)
    if fmt == "historical-basic":
        return normalize_historical_basic(odds)
    if fmt == "oddsportal":
        return normalize_oddsportal(odds)
    return odds


def fill_from_secondary(primary_attached: pd.DataFrame, secondary_attached: pd.DataFrame) -> pd.DataFrame:
    out = primary_attached.copy()
    fill_columns = [
        "sportsbook",
        "total_line",
        "total_line_open",
        "total_line_move",
        "total_line_source",
        "total_price_over",
        "total_price_under",
        "total_price_over_open",
        "total_price_under_open",
        "home_moneyline",
        "away_moneyline",
        "home_moneyline_open",
        "away_moneyline_open",
        "home_spread",
        "away_spread",
        "home_spread_price",
        "away_spread_price",
        "market_timestamp_utc",
        "historical_odds_game_id",
    ]
    for col in fill_columns:
        if col in secondary_attached.columns:
            if col not in out.columns:
                out[col] = secondary_attached[col]
            else:
                out[col] = out[col].combine_first(secondary_attached[col])
    if "total_line" in primary_attached.columns and "total_line" in secondary_attached.columns:
        out["odds_source_priority"] = "primary"
        filled_by_secondary = primary_attached["total_line"].isna() & secondary_attached["total_line"].notna()
        out.loc[filled_by_secondary, "odds_source_priority"] = "secondary"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach sportsbook total lines to MLB predictions.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--odds", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--games", type=Path, help="Optional MLB Stats API games CSV for date/team to game_pk joins.")
    parser.add_argument(
        "--format",
        choices=[
            "standard",
            "sports-odds-history",
            "historical-basic",
            "sbr-json",
            "oddsportal",
            "oddschecker-html",
            "the-odds-api-json",
            "api-sports-baseball-json",
        ],
        default="standard",
        help="Use historical-basic, sbr-json, the-odds-api-json, sports-odds-history, oddsportal, or standard normalized CSV.",
    )
    parser.add_argument("--matched-odds-out", type=Path, help="Optional path to write normalized odds with game_pk.")
    parser.add_argument("--assume-odds-date", help="Fill game_date for current odds pages that only show a display date.")
    parser.add_argument(
        "--sportsbook",
        default="consensus",
        help="For JSON odds: consensus, all, or a sportsbook key like fanduel or draftkings.",
    )
    parser.add_argument("--fetch-the-odds-api", action="store_true", help="Fetch current MLB odds from The Odds API v4.")
    parser.add_argument("--the-odds-api-key", default=os.getenv("THE_ODDS_API_KEY"), help="Defaults to THE_ODDS_API_KEY.")
    parser.add_argument("--the-odds-api-regions", default="us")
    parser.add_argument("--the-odds-api-markets", default="h2h,totals")
    parser.add_argument("--the-odds-api-bookmakers", help="Optional comma-delimited bookmaker keys.")
    parser.add_argument("--the-odds-api-odds-format", choices=["american", "decimal"], default="american")
    parser.add_argument(
        "--the-odds-api-date",
        help="Optional ISO8601 historical snapshot date, for example 2026-05-01T16:00:00Z. Requires historical API access.",
    )
    parser.add_argument("--the-odds-api-raw-out", type=Path, help="Optional path to cache the raw API JSON.")
    parser.add_argument("--fetch-api-sports-baseball", action="store_true", help="Fetch baseball odds from API-Sports/API-Football Baseball.")
    parser.add_argument("--api-sports-key", default=os.getenv("API_SPORTS_KEY") or os.getenv("APISPORTS_KEY"))
    parser.add_argument("--api-sports-base-url", default=API_SPORTS_BASEBALL_BASE_URL)
    parser.add_argument("--api-sports-date", help="Date filter, e.g. 2026-05-01.")
    parser.add_argument("--api-sports-game", help="Optional provider game id.")
    parser.add_argument("--api-sports-league", help="Optional league id.")
    parser.add_argument("--api-sports-season", help="Optional season.")
    parser.add_argument("--api-sports-bookmaker", help="Optional bookmaker id.")
    parser.add_argument("--api-sports-bet", help="Optional bet/market id.")
    parser.add_argument("--api-sports-raw-out", type=Path, help="Optional path to cache raw API-Sports JSON.")
    parser.add_argument("--secondary-odds", type=Path, help="Optional secondary odds file to fill missing primary lines.")
    parser.add_argument(
        "--secondary-format",
        choices=[
            "standard",
            "sports-odds-history",
            "historical-basic",
            "sbr-json",
            "oddsportal",
            "oddschecker-html",
            "the-odds-api-json",
            "api-sports-baseball-json",
        ],
        default="oddsportal",
    )
    parser.add_argument("--secondary-sportsbook", default="consensus")
    parser.add_argument("--secondary-matched-odds-out", type=Path)
    parser.add_argument("--fetch-oddschecker", action="store_true", help="Fetch current OddsChecker MLB odds using a headless browser.")
    parser.add_argument("--oddschecker-url", default=ODDSCHECKER_MLB_URL)
    parser.add_argument("--oddschecker-browser-path", help="Optional explicit Chrome/Edge executable path.")
    parser.add_argument("--oddschecker-raw-out", type=Path, default=Path("data/raw/oddschecker_mlb_current.html"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    if args.fetch_api_sports_baseball:
        if not args.api_sports_key:
            raise ValueError("Set API_SPORTS_KEY/APISPORTS_KEY or pass --api-sports-key.")
        payload = fetch_api_sports_baseball_json(
            args.api_sports_key,
            base_url=args.api_sports_base_url,
            date=args.api_sports_date,
            game=args.api_sports_game,
            league=args.api_sports_league,
            season=args.api_sports_season,
            bookmaker=args.api_sports_bookmaker,
            bet=args.api_sports_bet,
        )
        if args.api_sports_raw_out:
            args.api_sports_raw_out.parent.mkdir(parents=True, exist_ok=True)
            args.api_sports_raw_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        odds = normalize_api_sports_baseball_json(payload, sportsbook=args.sportsbook)
        odds_format = "api-sports-baseball-json"
    elif args.fetch_oddschecker:
        fetch_oddschecker_html_with_browser(
            args.oddschecker_raw_out,
            url=args.oddschecker_url,
            browser_path=args.oddschecker_browser_path,
        )
        odds = normalize_oddschecker_html(args.oddschecker_raw_out)
        odds_format = "oddschecker-html"
    elif args.fetch_the_odds_api:
        if not args.the_odds_api_key:
            raise ValueError("Set THE_ODDS_API_KEY or pass --the-odds-api-key to fetch current odds.")
        payload = fetch_the_odds_api_json(
            args.the_odds_api_key,
            regions=args.the_odds_api_regions,
            markets=args.the_odds_api_markets,
            odds_format=args.the_odds_api_odds_format,
            bookmakers=args.the_odds_api_bookmakers,
            snapshot_date=args.the_odds_api_date,
        )
        if args.the_odds_api_raw_out:
            args.the_odds_api_raw_out.parent.mkdir(parents=True, exist_ok=True)
            args.the_odds_api_raw_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        odds = normalize_the_odds_api_json(payload, sportsbook=args.sportsbook)
        odds_format = "the-odds-api-json"
    else:
        if not args.odds:
            raise ValueError("Pass --odds or use --fetch-the-odds-api.")
        odds = load_odds(args.odds, fmt=args.format, sportsbook=args.sportsbook)
        odds_format = args.format
    if odds_format in {
        "sports-odds-history",
        "historical-basic",
        "sbr-json",
        "oddsportal",
        "oddschecker-html",
        "the-odds-api-json",
        "api-sports-baseball-json",
    }:
        if args.assume_odds_date and "game_date" not in odds.columns:
            odds["game_date"] = pd.to_datetime(args.assume_odds_date, errors="raise").date().isoformat()
        if args.games:
            odds = attach_game_ids(pd.read_csv(args.games), odds)
        if args.matched_odds_out:
            args.matched_odds_out.parent.mkdir(parents=True, exist_ok=True)
            odds.to_csv(args.matched_odds_out, index=False)
    out = attach_totals(predictions, odds)
    if args.secondary_odds:
        secondary = load_odds(args.secondary_odds, fmt=args.secondary_format, sportsbook=args.secondary_sportsbook)
        if args.secondary_format in {
            "sports-odds-history",
            "historical-basic",
            "sbr-json",
            "oddsportal",
            "oddschecker-html",
            "the-odds-api-json",
            "api-sports-baseball-json",
        } and args.games:
            secondary = attach_game_ids(pd.read_csv(args.games), secondary)
        if args.secondary_matched_odds_out:
            args.secondary_matched_odds_out.parent.mkdir(parents=True, exist_ok=True)
            secondary.to_csv(args.secondary_matched_odds_out, index=False)
        out = fill_from_secondary(out, attach_totals(predictions, secondary))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    missing = int(out["total_line"].isna().sum()) if "total_line" in out.columns else len(out)
    print(f"Wrote {len(out)} rows to {args.out}; missing total_line={missing}")


if __name__ == "__main__":
    main()
