from __future__ import annotations

import argparse
import http.client
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .governance import canonical_hash, utc_now_iso, write_manifest


SAVANT_STATCAST_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"

SWINGING_STRIKES = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
CALLED_STRIKES = {"called_strike"}
FASTBALLS = {"FF", "SI", "FC", "FA"}
BREAKING = {"SL", "CU", "KC", "ST", "SV"}
OFFSPEED = {"CH", "FS", "FO", "SC"}

STATCAST_FEATURE_COLUMNS = [
    "sp_statcast_pitches_prior",
    "sp_statcast_bip_prior",
    "sp_statcast_avg_velocity_prior",
    "sp_statcast_avg_spin_prior",
    "sp_statcast_whiff_rate_prior",
    "sp_statcast_csw_rate_prior",
    "sp_statcast_zone_rate_prior",
    "sp_statcast_barrel_rate_prior",
    "sp_statcast_hard_hit_rate_prior",
    "sp_statcast_xwoba_allowed_prior",
    "sp_statcast_k_rate_prior",
    "sp_statcast_bb_rate_prior",
    "sp_statcast_hr_rate_prior",
    "sp_statcast_fastball_pct_prior",
    "sp_statcast_breaking_pct_prior",
    "sp_statcast_offspeed_pct_prior",
    "sp_statcast_rhb_pct_prior",
    "sp_statcast_lhb_pct_prior",
]


def _statcast_url(start_date: date, end_date: date) -> str:
    params = {
        "all": "true",
        "hfPT": "",
        "hfAB": "",
        "hfGT": "R|",
        "hfPR": "",
        "hfZ": "",
        "stadium": "",
        "hfBBL": "",
        "hfNewZones": "",
        "hfPull": "",
        "hfC": "",
        "hfSea": "",
        "hfSit": "",
        "player_type": "pitcher",
        "hfOuts": "",
        "opponent": "",
        "pitcher_throws": "",
        "batter_stands": "",
        "hfSA": "",
        "game_date_gt": start_date.isoformat(),
        "game_date_lt": end_date.isoformat(),
        "team": "",
        "position": "",
        "hfRO": "",
        "home_road": "",
        "hfFlag": "",
        "hfInfield": "",
        "metric_1": "",
        "group_by": "name",
        "min_pitches": "0",
        "min_results": "0",
        "min_pas": "0",
        "sort_col": "pitches",
        "player_event_sort": "h_launch_speed",
        "sort_order": "desc",
        "type": "details",
    }
    return f"{SAVANT_STATCAST_CSV}?{urlencode(params)}"


def _chunk_dates(start_date: date, end_date: date, days: int) -> list[tuple[date, date]]:
    chunks = []
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=days - 1), end_date)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def fetch_statcast_csv(
    start_date: date,
    end_date: date,
    *,
    raw_dir: Path,
    chunk_days: int = 7,
    force: bool = False,
    retries: int = 3,
) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for chunk_start, chunk_end in _chunk_dates(start_date, end_date, chunk_days):
        path = raw_dir / f"statcast_{chunk_start.isoformat()}_{chunk_end.isoformat()}.csv"
        if force or not path.exists():
            request = Request(
                _statcast_url(chunk_start, chunk_end),
                headers={"User-Agent": "Mozilla/5.0 mlb-model-statcast/1.0"},
            )
            last_error: Exception | None = None
            for attempt in range(retries):
                try:
                    with urlopen(request, timeout=180) as response:
                        payload = response.read()
                    break
                except (URLError, TimeoutError, http.client.IncompleteRead) as exc:
                    last_error = exc
                    if attempt == retries - 1:
                        raise
                    time.sleep(3.0 * (attempt + 1))
            else:
                raise RuntimeError(f"Statcast fetch failed for {chunk_start} to {chunk_end}") from last_error
            path.write_bytes(payload)
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def fetch_pitcher_daily_statcast_features(
    start_date: date,
    end_date: date,
    *,
    raw_dir: Path,
    chunk_days: int = 7,
    force: bool = False,
) -> pd.DataFrame:
    frames = []
    for chunk_start, chunk_end in _chunk_dates(start_date, end_date, chunk_days):
        statcast = fetch_statcast_csv(
            chunk_start,
            chunk_end,
            raw_dir=raw_dir,
            chunk_days=chunk_days,
            force=force,
        )
        aggregates = build_pitcher_daily_statcast_aggregates(statcast)
        if not aggregates.empty:
            frames.append(aggregates)
    if not frames:
        return pd.DataFrame(columns=["pitcher", "game_date"] + STATCAST_FEATURE_COLUMNS)
    aggregates = pd.concat(frames, ignore_index=True, sort=False)
    sum_cols = [col for col in aggregates.columns if col not in {"pitcher", "game_date"}]
    aggregates = aggregates.groupby(["pitcher", "game_date"], as_index=False, sort=True)[sum_cols].sum(min_count=1)
    return prior_from_pitcher_daily_statcast_aggregates(aggregates)


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out


def _barrel_mask(frame: pd.DataFrame) -> pd.Series:
    if "launch_speed_angle" in frame.columns:
        return pd.to_numeric(frame["launch_speed_angle"], errors="coerce").eq(6)
    speed = pd.to_numeric(frame.get("launch_speed"), errors="coerce")
    angle = pd.to_numeric(frame.get("launch_angle"), errors="coerce")
    return speed.ge(98) & angle.between(26, 30)


def build_pitcher_daily_statcast_aggregates(statcast: pd.DataFrame) -> pd.DataFrame:
    if statcast.empty:
        return pd.DataFrame()

    needed = [
        "game_date",
        "game_pk",
        "pitcher",
        "pitch_type",
        "description",
        "type",
        "zone",
        "events",
        "stand",
        "release_speed",
        "release_spin_rate",
        "estimated_woba_using_speedangle",
        "woba_value",
        "launch_speed",
        "launch_angle",
        "launch_speed_angle",
    ]
    raw = _ensure_columns(statcast, needed)
    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce").dt.date.astype(str)
    raw["pitcher"] = pd.to_numeric(raw["pitcher"], errors="coerce")
    raw = raw.dropna(subset=["game_date", "pitcher"]).copy()
    raw["pitcher"] = raw["pitcher"].astype(int)
    raw["pitch_count"] = 1
    description = raw["description"].fillna("").astype(str)
    pitch_type = raw["pitch_type"].fillna("").astype(str)
    events = raw["events"].fillna("").astype(str)
    raw["whiff"] = description.isin(SWINGING_STRIKES).astype(int)
    raw["csw"] = (description.isin(SWINGING_STRIKES | CALLED_STRIKES)).astype(int)
    raw["zone_pitch"] = pd.to_numeric(raw["zone"], errors="coerce").between(1, 9).astype(int)
    raw["bip"] = pd.to_numeric(raw["launch_speed"], errors="coerce").notna().astype(int)
    raw["barrel"] = _barrel_mask(raw).astype(int)
    raw["hard_hit"] = pd.to_numeric(raw["launch_speed"], errors="coerce").ge(95).astype(int)
    raw["xwoba_bip"] = pd.to_numeric(raw["estimated_woba_using_speedangle"], errors="coerce")
    raw["woba_event"] = pd.to_numeric(raw["woba_value"], errors="coerce")
    raw["k_event"] = events.eq("strikeout").astype(int)
    raw["bb_event"] = events.isin(["walk", "intent_walk"]).astype(int)
    raw["hr_event"] = events.eq("home_run").astype(int)
    raw["fastball"] = pitch_type.isin(FASTBALLS).astype(int)
    raw["breaking"] = pitch_type.isin(BREAKING).astype(int)
    raw["offspeed"] = pitch_type.isin(OFFSPEED).astype(int)
    raw["rhb"] = raw["stand"].fillna("").astype(str).eq("R").astype(int)
    raw["lhb"] = raw["stand"].fillna("").astype(str).eq("L").astype(int)

    grouped = (
        raw.groupby(["pitcher", "game_date"], sort=True)
        .agg(
            pitches=("pitch_count", "sum"),
            bip=("bip", "sum"),
            release_speed_sum=("release_speed", "sum"),
            release_spin_sum=("release_spin_rate", "sum"),
            whiffs=("whiff", "sum"),
            csw=("csw", "sum"),
            zone=("zone_pitch", "sum"),
            barrels=("barrel", "sum"),
            hard_hits=("hard_hit", "sum"),
            xwoba_sum=("xwoba_bip", "sum"),
            xwoba_count=("xwoba_bip", "count"),
            woba_sum=("woba_event", "sum"),
            woba_count=("woba_event", "count"),
            strikeouts=("k_event", "sum"),
            walks=("bb_event", "sum"),
            homers=("hr_event", "sum"),
            fastballs=("fastball", "sum"),
            breaking=("breaking", "sum"),
            offspeed=("offspeed", "sum"),
            rhb=("rhb", "sum"),
            lhb=("lhb", "sum"),
        )
        .reset_index()
    )
    return grouped


def prior_from_pitcher_daily_statcast_aggregates(grouped: pd.DataFrame) -> pd.DataFrame:
    if grouped.empty:
        return pd.DataFrame(columns=["pitcher", "game_date"] + STATCAST_FEATURE_COLUMNS)
    grouped = grouped.copy()
    grouped["game_date_dt"] = pd.to_datetime(grouped["game_date"], errors="raise")
    grouped = grouped.sort_values(["pitcher", "game_date_dt"], kind="mergesort").reset_index(drop=True)
    cumulative_cols = [
        "pitches",
        "bip",
        "release_speed_sum",
        "release_spin_sum",
        "whiffs",
        "csw",
        "zone",
        "barrels",
        "hard_hits",
        "xwoba_sum",
        "xwoba_count",
        "woba_sum",
        "woba_count",
        "strikeouts",
        "walks",
        "homers",
        "fastballs",
        "breaking",
        "offspeed",
        "rhb",
        "lhb",
    ]
    prior = grouped[["pitcher", "game_date"]].copy()
    for col in cumulative_cols:
        prior[col] = grouped.groupby("pitcher", sort=False)[col].cumsum().groupby(grouped["pitcher"], sort=False).shift(1)
    denom = prior["pitches"].replace(0, np.nan)
    bip = prior["bip"].replace(0, np.nan)
    xwoba_count = prior["xwoba_count"].replace(0, np.nan)
    woba_count = prior["woba_count"].replace(0, np.nan)
    prior["sp_statcast_pitches_prior"] = prior["pitches"]
    prior["sp_statcast_bip_prior"] = prior["bip"]
    prior["sp_statcast_avg_velocity_prior"] = prior["release_speed_sum"] / denom
    prior["sp_statcast_avg_spin_prior"] = prior["release_spin_sum"] / denom
    prior["sp_statcast_whiff_rate_prior"] = prior["whiffs"] / denom
    prior["sp_statcast_csw_rate_prior"] = prior["csw"] / denom
    prior["sp_statcast_zone_rate_prior"] = prior["zone"] / denom
    prior["sp_statcast_barrel_rate_prior"] = prior["barrels"] / bip
    prior["sp_statcast_hard_hit_rate_prior"] = prior["hard_hits"] / bip
    prior["sp_statcast_xwoba_allowed_prior"] = (prior["xwoba_sum"] / xwoba_count).combine_first(prior["woba_sum"] / woba_count)
    prior["sp_statcast_k_rate_prior"] = prior["strikeouts"] / denom
    prior["sp_statcast_bb_rate_prior"] = prior["walks"] / denom
    prior["sp_statcast_hr_rate_prior"] = prior["homers"] / denom
    prior["sp_statcast_fastball_pct_prior"] = prior["fastballs"] / denom
    prior["sp_statcast_breaking_pct_prior"] = prior["breaking"] / denom
    prior["sp_statcast_offspeed_pct_prior"] = prior["offspeed"] / denom
    prior["sp_statcast_rhb_pct_prior"] = prior["rhb"] / denom
    prior["sp_statcast_lhb_pct_prior"] = prior["lhb"] / denom
    return prior[["pitcher", "game_date"] + STATCAST_FEATURE_COLUMNS].round(6)


def build_pitcher_daily_statcast_features(statcast: pd.DataFrame) -> pd.DataFrame:
    return prior_from_pitcher_daily_statcast_aggregates(build_pitcher_daily_statcast_aggregates(statcast))


def attach_statcast_starter_features(features: pd.DataFrame, pitcher_daily: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    if pitcher_daily.empty:
        out["statcast_feature_source"] = "missing_statcast_pitch_data"
        return out
    daily = pitcher_daily.copy()
    daily["game_date"] = pd.to_datetime(daily["game_date"], errors="raise").dt.date.astype(str)
    for side in ["home", "away"]:
        starter_col = f"{side}_actual_starter_id"
        if starter_col not in out.columns:
            continue
        right = daily.rename(
            columns={
                "pitcher": starter_col,
                **{col: f"{side}_{col}" for col in STATCAST_FEATURE_COLUMNS},
            }
        )
        out["game_date"] = pd.to_datetime(out["game_date"], errors="raise").dt.date.astype(str)
        out[starter_col] = pd.to_numeric(out[starter_col], errors="coerce")
        right[starter_col] = pd.to_numeric(right[starter_col], errors="coerce")
        out = out.merge(right, on=[starter_col, "game_date"], how="left")

    if {"home_sp_statcast_xwoba_allowed_prior", "away_sp_statcast_xwoba_allowed_prior"}.issubset(out.columns):
        out["sp_statcast_xwoba_diff"] = out["home_sp_statcast_xwoba_allowed_prior"] - out["away_sp_statcast_xwoba_allowed_prior"]
        out["sp_statcast_whiff_diff"] = out["home_sp_statcast_whiff_rate_prior"] - out["away_sp_statcast_whiff_rate_prior"]
        out["sp_statcast_csw_diff"] = out["home_sp_statcast_csw_rate_prior"] - out["away_sp_statcast_csw_rate_prior"]
        out["sp_statcast_hard_hit_diff"] = out["home_sp_statcast_hard_hit_rate_prior"] - out["away_sp_statcast_hard_hit_rate_prior"]
        out["sp_statcast_barrel_diff"] = out["home_sp_statcast_barrel_rate_prior"] - out["away_sp_statcast_barrel_rate_prior"]
        out["sp_statcast_velocity_diff"] = out["home_sp_statcast_avg_velocity_prior"] - out["away_sp_statcast_avg_velocity_prior"]
        out["sp_statcast_pitch_mix_gap"] = (
            (out["home_sp_statcast_fastball_pct_prior"] - out["away_sp_statcast_fastball_pct_prior"]).abs()
            + (out["home_sp_statcast_breaking_pct_prior"] - out["away_sp_statcast_breaking_pct_prior"]).abs()
            + (out["home_sp_statcast_offspeed_pct_prior"] - out["away_sp_statcast_offspeed_pct_prior"]).abs()
        )
        out["statcast_feature_source"] = np.where(
            out[["home_sp_statcast_pitches_prior", "away_sp_statcast_pitches_prior"]].notna().all(axis=1),
            "baseball_savant_statcast_rolling_prior",
            "partial_or_missing_statcast_pitch_data",
        )
    else:
        out["statcast_feature_source"] = "missing_actual_starter_ids"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and attach Baseball Savant Statcast starter features.")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--statcast-csv", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--pitcher-daily-out", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--chunk-days", default=7, type=int)
    parser.add_argument("--force-fetch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.statcast_csv:
        statcast = pd.read_csv(args.statcast_csv, low_memory=False)
    else:
        if not args.start_date or not args.end_date or not args.raw_dir:
            raise ValueError("Provide --statcast-csv, or --start-date/--end-date/--raw-dir")
        statcast = fetch_statcast_csv(
            args.start_date,
            args.end_date,
            raw_dir=args.raw_dir,
            chunk_days=args.chunk_days,
            force=args.force_fetch,
        )
    daily = build_pitcher_daily_statcast_features(statcast)
    if args.pitcher_daily_out:
        args.pitcher_daily_out.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(args.pitcher_daily_out, index=False)
    outputs = {"pitcher_daily_rows": len(daily), "statcast_rows": len(statcast)}
    if args.features and args.out:
        features = pd.read_csv(args.features, low_memory=False)
        enriched = attach_statcast_starter_features(features, daily)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(args.out, index=False)
        outputs.update({"features_hash": canonical_hash(features), "output_hash": canonical_hash(enriched), "output_rows": len(enriched)})
    if args.out:
        write_manifest(
            args.out.with_suffix(".manifest.json"),
            {"generated_at_utc": utc_now_iso(), "source": "baseball_savant_statcast", **outputs},
        )
    elif args.pitcher_daily_out:
        write_manifest(
            args.pitcher_daily_out.with_suffix(".manifest.json"),
            {"generated_at_utc": utc_now_iso(), "source": "baseball_savant_statcast", **outputs},
        )
    print(outputs)


if __name__ == "__main__":
    main()
