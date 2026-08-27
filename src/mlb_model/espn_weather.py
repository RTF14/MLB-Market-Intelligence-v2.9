from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .governance import canonical_hash, utc_now_iso, write_manifest


ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"


TEAM_ALIASES = {
    "arizona diamondbacks": "ARI",
    "atlanta braves": "ATL",
    "baltimore orioles": "BAL",
    "boston red sox": "BOS",
    "chicago cubs": "CHC",
    "chicago white sox": "CHW",
    "cincinnati reds": "CIN",
    "cleveland guardians": "CLE",
    "colorado rockies": "COL",
    "detroit tigers": "DET",
    "houston astros": "HOU",
    "kansas city royals": "KC",
    "los angeles angels": "LAA",
    "los angeles dodgers": "LAD",
    "miami marlins": "MIA",
    "milwaukee brewers": "MIL",
    "minnesota twins": "MIN",
    "new york mets": "NYM",
    "new york yankees": "NYY",
    "athletics": "ATH",
    "oakland athletics": "ATH",
    "sacramento athletics": "ATH",
    "philadelphia phillies": "PHI",
    "pittsburgh pirates": "PIT",
    "san diego padres": "SD",
    "san francisco giants": "SF",
    "seattle mariners": "SEA",
    "st. louis cardinals": "STL",
    "st louis cardinals": "STL",
    "tampa bay rays": "TB",
    "texas rangers": "TEX",
    "toronto blue jays": "TOR",
    "washington nationals": "WSH",
}


def _normalize_team(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text in TEAM_ALIASES:
        return TEAM_ALIASES[text]
    parts = text.split()
    return (parts[-1][:3] if parts else "").upper()


def _num(value: object) -> float | None:
    if value in [None, ""]:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _weather_value(weather: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in weather and weather.get(name) not in [None, ""]:
            return weather.get(name)
    return None


def _extract_weather(event: dict[str, Any]) -> dict[str, Any]:
    competition = (event.get("competitions") or [{}])[0]
    weather = competition.get("weather") or event.get("weather") or {}
    return {
        "temperature_f": _num(_weather_value(weather, ["temperature", "temperatureF", "temp", "displayTemperature"])),
        "wind_mph": _num(_weather_value(weather, ["windSpeed", "wind_mph", "windSpeedMph", "displayWindSpeed"])),
        "humidity_pct": _num(_weather_value(weather, ["humidity", "relativeHumidity", "humidityPct"])),
        "weather_condition": _weather_value(weather, ["condition", "displayValue", "description", "shortDisplayValue"]),
        "wind_direction": _weather_value(weather, ["windDirection", "windDirectionAbbreviation", "displayWindDirection"]),
        "weather_raw": json.dumps(weather, sort_keys=True, default=str) if weather else "",
    }


@dataclass(frozen=True)
class EspnWeatherClient:
    sleep_seconds: float = 0.15
    timeout_seconds: int = 30

    def scoreboard(self, game_date: date | str) -> dict[str, Any]:
        params = {"dates": str(game_date).replace("-", ""), "limit": 100}
        request = Request(
            f"{ESPN_SCOREBOARD_URL}?{urlencode(params)}",
            headers={"User-Agent": "Mozilla/5.0 mlb-model-weather/1.0"},
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return payload


def scoreboard_weather_rows(payload: dict[str, Any], fallback_date: date | str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        competition = (event.get("competitions") or [{}])[0]
        competitors = competition.get("competitors") or []
        home = next((item for item in competitors if item.get("homeAway") == "home"), {})
        away = next((item for item in competitors if item.get("homeAway") == "away"), {})
        home_team = home.get("team") or {}
        away_team = away.get("team") or {}
        weather = _extract_weather(event)
        espn_event_date_utc = str(event.get("date") or competition.get("date") or "")[:10]
        # ESPN event dates are UTC; MLB schedule dates are local baseball dates.
        # Use the requested scoreboard date for joins and preserve ESPN's UTC date separately.
        game_date = str(fallback_date or espn_event_date_utc or "")[:10]
        rows.append(
            {
                "game_date": game_date,
                "espn_event_date_utc": espn_event_date_utc,
                "espn_event_id": event.get("id"),
                "espn_uid": event.get("uid"),
                "home_team": home_team.get("displayName") or home_team.get("name"),
                "away_team": away_team.get("displayName") or away_team.get("name"),
                "home_team_abbr": home_team.get("abbreviation") or _normalize_team(home_team.get("displayName")),
                "away_team_abbr": away_team.get("abbreviation") or _normalize_team(away_team.get("displayName")),
                "espn_status": (competition.get("status") or {}).get("type", {}).get("description"),
                "weather_source": "espn_scoreboard",
                **weather,
            }
        )
    return rows


def fetch_weather(start_date: date, end_date: date, *, raw_dir: Path | None = None) -> pd.DataFrame:
    client = EspnWeatherClient()
    rows: list[dict[str, Any]] = []
    current = start_date
    while current <= end_date:
        payload = client.scoreboard(current)
        if raw_dir:
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"espn_scoreboard_{current.isoformat()}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        rows.extend(scoreboard_weather_rows(payload, current))
        current += timedelta(days=1)
    return pd.DataFrame(rows).sort_values(["game_date", "espn_event_id"]).reset_index(drop=True)


def attach_weather(games: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    required = {"game_date", "home_team", "away_team"}
    missing_games = required - set(games.columns)
    missing_weather = required - set(weather.columns)
    if missing_games:
        raise ValueError(f"Game frame missing columns for ESPN weather join: {sorted(missing_games)}")
    if missing_weather:
        raise ValueError(f"Weather frame missing columns for ESPN weather join: {sorted(missing_weather)}")

    left = games.copy()
    right = weather.copy()
    left["game_date"] = pd.to_datetime(left["game_date"], errors="raise").dt.date.astype(str)
    right["game_date"] = pd.to_datetime(right["game_date"], errors="coerce").dt.date.astype(str)
    left["_home_key"] = left["home_team"].map(_normalize_team)
    left["_away_key"] = left["away_team"].map(_normalize_team)
    right["_home_key"] = right["home_team"].map(_normalize_team)
    right["_away_key"] = right["away_team"].map(_normalize_team)
    weather_cols = [
        "game_date",
        "_home_key",
        "_away_key",
        "espn_event_id",
        "temperature_f",
        "wind_mph",
        "humidity_pct",
        "weather_condition",
        "wind_direction",
        "weather_source",
        "weather_raw",
    ]
    available = [col for col in weather_cols if col in right.columns]
    out = left.merge(right[available], on=["game_date", "_home_key", "_away_key"], how="left", suffixes=("", "_espn"))
    out["weather_join_matched"] = out["espn_event_id"].notna()
    return out.drop(columns=["_home_key", "_away_key"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ESPN MLB scoreboard weather and optionally join it to MLB game rows.")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--weather", type=Path, help="Existing ESPN weather CSV to join.")
    parser.add_argument("--games", type=Path, help="Optional game CSV to enrich with weather.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--raw-dir", type=Path, help="Optional directory for raw ESPN scoreboard JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.weather:
        weather = pd.read_csv(args.weather)
    else:
        if not args.start_date or not args.end_date:
            raise ValueError("Provide --start-date and --end-date when --weather is not supplied")
        weather = fetch_weather(args.start_date, args.end_date, raw_dir=args.raw_dir)

    if args.games:
        games = pd.read_csv(args.games)
        output = attach_weather(games, weather)
    else:
        output = weather

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    write_manifest(
        args.out.with_suffix(".manifest.json"),
        {
            "generated_at_utc": utc_now_iso(),
            "source": ESPN_SCOREBOARD_URL,
            "rows": len(output),
            "weather_rows": len(weather),
            "joined_to_games": bool(args.games),
            "output_hash": canonical_hash(output),
        },
    )
    print(f"Wrote {len(output)} rows to {args.out}")


if __name__ == "__main__":
    main()
