from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .governance import canonical_hash, utc_now_iso, write_manifest


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

VENUE_COORDS = {
    1: ("Angel Stadium", 33.8003, -117.8827),
    2: ("Oriole Park at Camden Yards", 39.2840, -76.6217),
    3: ("Fenway Park", 42.3467, -71.0972),
    4: ("Guaranteed Rate Field", 41.8300, -87.6339),
    5: ("Progressive Field", 41.4962, -81.6852),
    7: ("Kauffman Stadium", 39.0517, -94.4803),
    10: ("Oakland Coliseum", 37.7516, -122.2005),
    12: ("Tropicana Field", 27.7682, -82.6534),
    14: ("Rogers Centre", 43.6414, -79.3894),
    15: ("Chase Field", 33.4455, -112.0667),
    17: ("Wrigley Field", 41.9484, -87.6553),
    19: ("Coors Field", 39.7559, -104.9942),
    22: ("Dodger Stadium", 34.0739, -118.2400),
    2602: ("American Family Field", 43.0280, -87.9712),
    2603: ("Great American Ball Park", 39.0979, -84.5066),
    2392: ("Minute Maid Park", 29.7573, -95.3555),
    2394: ("Comerica Park", 42.3390, -83.0485),
    2395: ("loanDepot park", 25.7781, -80.2197),
    2396: ("Yankee Stadium", 40.8296, -73.9262),
    2397: ("Citizens Bank Park", 39.9061, -75.1665),
    2398: ("PNC Park", 40.4469, -80.0057),
    2399: ("Busch Stadium", 38.6226, -90.1928),
    2530: ("Petco Park", 32.7073, -117.1566),
    2393: ("T-Mobile Park", 47.5914, -122.3325),
    2390: ("Oracle Park", 37.7786, -122.3893),
    31: ("PNC Park", 40.4469, -80.0057),
    32: ("American Family Field", 43.0280, -87.9712),
    680: ("T-Mobile Park", 47.5914, -122.3325),
    2529: ("Sutter Health Park", 38.5804, -121.5132),
    2680: ("Petco Park", 32.7073, -117.1566),
    2681: ("Citizens Bank Park", 39.9061, -75.1665),
    2889: ("Busch Stadium", 38.6226, -90.1928),
    3289: ("Nationals Park", 38.8730, -77.0074),
    3309: ("Nationals Park", 38.8730, -77.0074),
    3312: ("Target Field", 44.9817, -93.2776),
    3313: ("Citi Field", 40.7571, -73.8458),
    3314: ("Truist Park", 33.8907, -84.4677),
    4169: ("loanDepot park", 25.7781, -80.2197),
    4705: ("Truist Park", 33.8907, -84.4677),
    5325: ("Globe Life Field", 32.7473, -97.0842),
    5355: ("Sutter Health Park", 38.5804, -121.5132),
}

ROOF_OR_DOME_VENUES = {12, 14, 15, 32, 2392, 2395, 2602, 4169, 5325}


def _weather_url(lat: float, lon: float, start_date: str, end_date: str) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation"]),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "UTC",
    }
    return f"{OPEN_METEO_ARCHIVE_URL}?{urlencode(params)}"


def fetch_venue_weather(
    venue_id: int,
    start_date: str,
    end_date: str,
    *,
    raw_dir: Path,
    force: bool = False,
    retries: int = 3,
) -> pd.DataFrame:
    if venue_id not in VENUE_COORDS:
        raise ValueError(f"No Open-Meteo coordinates configured for venue_id={venue_id}")
    name, lat, lon = VENUE_COORDS[venue_id]
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"open_meteo_venue_{venue_id}_{start_date}_{end_date}.json"
    if force or not path.exists():
        request = Request(_weather_url(lat, lon, start_date, end_date), headers={"User-Agent": "mlb-model-weather/1.0"})
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                with urlopen(request, timeout=90) as response:
                    payload = response.read().decode("utf-8")
                break
            except URLError as exc:
                last_error = exc
                if attempt == retries - 1:
                    raise
                time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"Open-Meteo fetch failed for venue_id={venue_id}") from last_error
        path.write_text(payload, encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    hourly = payload.get("hourly", {})
    frame = pd.DataFrame(hourly)
    if frame.empty:
        return pd.DataFrame()
    frame["venue_id"] = int(venue_id)
    frame["venue_name_weather"] = name
    frame["weather_latitude"] = lat
    frame["weather_longitude"] = lon
    frame["weather_source"] = "open_meteo_archive"
    return frame


def fetch_weather_for_games(games: pd.DataFrame, *, raw_dir: Path, force: bool = False) -> pd.DataFrame:
    required = {"venue_id", "game_date"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Game frame missing weather columns: {sorted(missing)}")
    work = games.copy()
    work["venue_id"] = pd.to_numeric(work["venue_id"], errors="coerce").astype("Int64")
    work["game_date"] = pd.to_datetime(work["game_date"], errors="raise").dt.date.astype(str)
    work["weather_fetch_date"] = _game_hour_utc(work).dt.date.astype(str)
    frames = []
    for venue_id, group in work.dropna(subset=["venue_id"]).groupby("venue_id", sort=True):
        venue = int(venue_id)
        if venue not in VENUE_COORDS:
            continue
        start = min(group["game_date"].min(), group["weather_fetch_date"].min())
        end = max(group["game_date"].max(), group["weather_fetch_date"].max())
        frames.append(fetch_venue_weather(venue, start, end, raw_dir=raw_dir, force=force))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _direction_bucket(degrees: pd.Series) -> pd.Series:
    deg = pd.to_numeric(degrees, errors="coerce")
    labels = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    idx = (((deg + 22.5) % 360) // 45).astype("Int64")
    out = pd.Series("UNKNOWN", index=degrees.index, dtype=object)
    valid = idx.notna()
    out.loc[valid] = labels[idx.loc[valid].astype(int).to_numpy()]
    return out


def _game_hour_utc(games: pd.DataFrame) -> pd.Series:
    for col in ["game_start_time_utc", "commence_time_utc", "game_datetime_utc"]:
        if col in games.columns:
            parsed = pd.to_datetime(games[col], utc=True, errors="coerce")
            if parsed.notna().any():
                fallback = pd.to_datetime(games["game_date"], utc=True, errors="raise") + pd.Timedelta(hours=23)
                return parsed.fillna(fallback).dt.floor("h")
    return (pd.to_datetime(games["game_date"], utc=True, errors="raise") + pd.Timedelta(hours=23)).dt.floor("h")


def attach_open_meteo_weather(games: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    out = games.copy()
    if weather.empty:
        out["open_meteo_join_matched"] = False
        return out
    out["venue_id"] = pd.to_numeric(out["venue_id"], errors="coerce").astype("Int64")
    out["weather_time_utc"] = _game_hour_utc(out)
    wx = weather.copy()
    wx["venue_id"] = pd.to_numeric(wx["venue_id"], errors="coerce").astype("Int64")
    wx["weather_time_utc"] = pd.to_datetime(wx["time"], utc=True, errors="coerce")
    cols = [
        "venue_id",
        "weather_time_utc",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "weather_source",
    ]
    wx = wx[[col for col in cols if col in wx.columns]].dropna(subset=["weather_time_utc"])
    merged = out.merge(wx, on=["venue_id", "weather_time_utc"], how="left", suffixes=("", "_open_meteo"))
    merged["open_meteo_join_matched"] = merged["temperature_2m"].notna()
    merged["temperature_open_meteo_f"] = pd.to_numeric(merged.get("temperature_2m"), errors="coerce")
    merged["humidity_open_meteo_pct"] = pd.to_numeric(merged.get("relative_humidity_2m"), errors="coerce")
    merged["wind_speed_10m_mph"] = pd.to_numeric(merged.get("wind_speed_10m"), errors="coerce")
    merged["wind_direction_10m_deg"] = pd.to_numeric(merged.get("wind_direction_10m"), errors="coerce")
    merged["precipitation_open_meteo_in"] = pd.to_numeric(merged.get("precipitation"), errors="coerce")
    merged["wind_direction_bucket"] = _direction_bucket(merged["wind_direction_10m_deg"])
    merged["dome_or_roof_flag"] = merged["venue_id"].astype("Int64").isin(ROOF_OR_DOME_VENUES).astype(int)
    temp_lift = (merged["temperature_open_meteo_f"].fillna(70.0) - 70.0) * 0.02
    wind_lift = merged["wind_speed_10m_mph"].fillna(0.0) * 0.035
    precip_drag = merged["precipitation_open_meteo_in"].fillna(0.0).clip(lower=0.0, upper=0.5) * -0.8
    merged["weather_run_index_open_meteo"] = (temp_lift + wind_lift + precip_drag).clip(-1.0, 1.75).round(3)
    merged.loc[merged["dome_or_roof_flag"].eq(1), "weather_run_index_open_meteo"] = 0.0
    merged["weather_directionality_source"] = "open_meteo_wind_direction_no_park_orientation"
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and attach Open-Meteo historical MLB weather.")
    parser.add_argument("--games", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--weather-out", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force-fetch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    games = pd.read_csv(args.games, low_memory=False)
    weather = fetch_weather_for_games(games, raw_dir=args.raw_dir, force=args.force_fetch)
    if args.weather_out:
        args.weather_out.parent.mkdir(parents=True, exist_ok=True)
        weather.to_csv(args.weather_out, index=False)
    enriched = attach_open_meteo_weather(games, weather)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.out, index=False)
    write_manifest(
        args.out.with_suffix(".manifest.json"),
        {
            "generated_at_utc": utc_now_iso(),
            "source": "open_meteo_archive",
            "games_hash": canonical_hash(games),
            "weather_hash": canonical_hash(weather),
            "output_hash": canonical_hash(enriched),
            "rows": len(enriched),
            "weather_rows": len(weather),
            "matched_rows": int(enriched["open_meteo_join_matched"].sum()),
        },
    )
    print({"rows": len(enriched), "weather_rows": len(weather), "matched_rows": int(enriched["open_meteo_join_matched"].sum())})


if __name__ == "__main__":
    main()
