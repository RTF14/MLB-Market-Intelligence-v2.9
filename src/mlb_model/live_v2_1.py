from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import pandas as pd

from .execution_v1_7 import (
    MLBExecutionV17Config,
    _file_hash,
    _load_json,
    _load_metadata,
    _stable_output_hash,
    _training_data_hash,
    execute_v1_7,
)
from .governance import canonical_hash, utc_now_iso, write_manifest
from .odds import attach_game_ids, attach_totals, fetch_the_odds_api_json, normalize_the_odds_api_json


LIVE_VERSION = "mlb_live_v2_1"


def _safe_timestamp(value: str) -> str:
    return value.replace(":", "").replace("-", "").replace(".", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch live MLB odds, attach them to predictions, and optionally execute v2.1.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--games", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("reports/live_v2_1"), type=Path)
    parser.add_argument("--attached-out", type=Path)
    parser.add_argument("--matched-odds-out", type=Path)
    parser.add_argument("--raw-odds-out", type=Path)
    parser.add_argument("--the-odds-api-key", default=os.getenv("THE_ODDS_API_KEY"))
    parser.add_argument("--regions", default="us")
    parser.add_argument("--markets", default="h2h,totals")
    parser.add_argument("--bookmakers")
    parser.add_argument(
        "--the-odds-api-date",
        help="Optional ISO8601 historical snapshot date, for example 2026-05-01T16:00:00Z. Requires historical API access.",
    )
    parser.add_argument("--odds-format", choices=["american", "decimal"], default="american")
    parser.add_argument("--sportsbook", default="consensus", help="consensus, all, or a sportsbook key like fanduel.")
    parser.add_argument("--run-execution", action="store_true")
    parser.add_argument("--ou-calibrator", type=Path)
    parser.add_argument("--capital-state", type=Path)
    parser.add_argument("--risk-state", type=Path)
    parser.add_argument("--previous-action")
    parser.add_argument("--disable-winner", action="store_true")
    parser.add_argument("--disable-market-ou", action="store_true")
    parser.add_argument("--winner-daily-cap", default=MLBExecutionV17Config.max_daily_winner_picks, type=int)
    parser.add_argument("--ou-daily-cap", default=MLBExecutionV17Config.max_daily_ou_picks, type=int)
    parser.add_argument("--min-ou-probability", default=MLBExecutionV17Config.min_calibrated_ou_probability, type=float)
    parser.add_argument("--min-ou-ev", default=MLBExecutionV17Config.min_calibrated_ou_ev, type=float)
    parser.add_argument("--max-vig", default=MLBExecutionV17Config.max_vig, type=float)
    parser.add_argument("--no-bet-edge-band", default=MLBExecutionV17Config.no_bet_edge_band, type=float)
    parser.add_argument("--require-market-timestamps", action="store_true")
    parser.add_argument("--max-market-data-age-minutes", default=MLBExecutionV17Config.max_market_data_age_minutes, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.the_odds_api_key:
        raise ValueError("Set THE_ODDS_API_KEY or pass --the-odds-api-key.")
    if args.run_execution and not args.ou_calibrator:
        raise ValueError("Pass --ou-calibrator when using --run-execution.")

    generated_at = utc_now_iso()
    stamp = _safe_timestamp(generated_at)
    run_dir = args.out_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_odds_out = args.raw_odds_out or run_dir / "the_odds_api_raw.json"
    matched_odds_out = args.matched_odds_out or run_dir / "the_odds_api_matched.csv"
    attached_out = args.attached_out or run_dir / "predictions_with_the_odds_api.csv"

    predictions = pd.read_csv(args.predictions)
    games = pd.read_csv(args.games)
    payload = fetch_the_odds_api_json(
        args.the_odds_api_key,
        regions=args.regions,
        markets=args.markets,
        odds_format=args.odds_format,
        bookmakers=args.bookmakers,
        snapshot_date=args.the_odds_api_date,
    )
    raw_odds_out.parent.mkdir(parents=True, exist_ok=True)
    raw_odds_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    odds = normalize_the_odds_api_json(payload, sportsbook=args.sportsbook)
    matched = attach_game_ids(games, odds)
    matched_odds_out.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(matched_odds_out, index=False)

    attached = attach_totals(predictions, matched)
    attached_out.parent.mkdir(parents=True, exist_ok=True)
    attached.to_csv(attached_out, index=False)

    execution_outputs: dict[str, str] = {}
    execution_dir = None
    cfg = None
    calibrator_metadata = None
    if args.run_execution:
        cfg = MLBExecutionV17Config(
            winner_enabled=not args.disable_winner,
            market_ou_enabled=not args.disable_market_ou,
            max_daily_winner_picks=args.winner_daily_cap,
            max_daily_ou_picks=args.ou_daily_cap,
            min_calibrated_ou_probability=args.min_ou_probability,
            min_calibrated_ou_ev=args.min_ou_ev,
            max_vig=args.max_vig,
            no_bet_edge_band=args.no_bet_edge_band,
            odds_format=args.odds_format,
            require_market_timestamps=args.require_market_timestamps,
            max_market_data_age_minutes=args.max_market_data_age_minutes,
        )
        outputs = execute_v1_7(
            attached,
            ou_calibrator_path=args.ou_calibrator,
            config=cfg,
            capital_state=_load_json(args.capital_state),
            risk_state=_load_json(args.risk_state),
            previous_action=args.previous_action,
        )
        execution_dir = run_dir / "execution_v1_7"
        execution_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in outputs.items():
            frame.to_csv(execution_dir / f"{name}.csv", index=False)
            execution_outputs[name] = _stable_output_hash(frame)
        calibrator_metadata = _load_metadata(args.ou_calibrator)

    write_manifest(
        run_dir / "manifest.json",
        {
            "live_version": LIVE_VERSION,
            "generated_at_utc": generated_at,
            "predictions": str(args.predictions),
            "games": str(args.games),
            "raw_odds_out": str(raw_odds_out),
            "matched_odds_out": str(matched_odds_out),
            "attached_out": str(attached_out),
            "execution_dir": str(execution_dir) if execution_dir else None,
            "the_odds_api": {
                "regions": args.regions,
                "markets": args.markets,
                "bookmakers": args.bookmakers,
                "historical_snapshot_date": args.the_odds_api_date,
                "odds_format": args.odds_format,
                "sportsbook": args.sportsbook,
                "events_returned": len(payload),
            },
            "hashes": {
                "predictions": canonical_hash(predictions),
                "games": canonical_hash(games),
                "matched_odds": canonical_hash(matched),
                "attached_predictions": canonical_hash(attached),
                "execution_outputs": execution_outputs,
            },
            "execution": {
                "ran": bool(args.run_execution),
                "config": asdict(cfg) if cfg else None,
                "ou_calibrator": str(args.ou_calibrator) if args.ou_calibrator else None,
                "ou_calibrator_file_hash": _file_hash(args.ou_calibrator) if args.ou_calibrator else None,
                "ou_calibrator_metadata_hash": canonical_hash(calibrator_metadata) if calibrator_metadata else None,
                "ou_calibrator_training_data_hash": _training_data_hash(calibrator_metadata) if calibrator_metadata else None,
            },
        },
    )

    missing_total = int(attached["total_line"].isna().sum()) if "total_line" in attached.columns else len(attached)
    print(f"Wrote v2.1 live run to {run_dir}")
    print(f"Attached predictions: {attached_out} rows={len(attached)} missing_total_line={missing_total}")
    if execution_dir:
        print(f"Execution outputs: {execution_dir}")


if __name__ == "__main__":
    main()
