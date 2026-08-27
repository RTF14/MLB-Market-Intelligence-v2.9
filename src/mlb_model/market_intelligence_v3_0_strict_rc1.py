from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .governance import canonical_hash, utc_now_iso, write_manifest
from .market_intelligence_v3_0 import (
    MarketIntelligenceV30Config,
    _summary_row_v30,
    run_v30,
)
from .market_snapshots import build_snapshot_tables


FROZEN_VERSION = "mlb_market_intelligence_v3_0_strict_rc1"


@dataclass(frozen=True)
class MarketIntelligenceV30StrictRC1Config(MarketIntelligenceV30Config):
    freeze_backfill_observe_only: bool = True
    backfill_live_stake_allowed: bool = False
    research_version: str = FROZEN_VERSION


def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[0:0].copy()


def _apply_strict_freeze(
    outputs: dict[str, pd.DataFrame],
    *,
    market: str,
    prefix: str,
    result_col: str,
) -> None:
    source = outputs.get(f"{prefix}_orders", pd.DataFrame()).copy()
    if source.empty:
        outputs[f"{prefix}_orders"] = source
        outputs[f"{prefix}_observe_only"] = source
        outputs[f"{prefix}_recommendations"] = source
        outputs[f"{prefix}_overall"] = pd.DataFrame([_summary_row_v30(source, "STRICT_EXECUTABLE", result_col)])
        outputs[f"{prefix}_tier_summary"] = _strict_tier_summary(source, market=market, result_col=result_col)
        return

    source["frozen_version"] = FROZEN_VERSION
    source["freeze_policy"] = "STRICT_EXECUTABLE_BACKFILL_OBSERVE_ONLY"
    source["shadow_stake_units"] = pd.to_numeric(source.get("stake_units", 0.0), errors="coerce").fillna(0.0)
    source["shadow_profit_units"] = pd.to_numeric(source.get("profit_units", 0.0), errors="coerce").fillna(0.0)
    source["order_intent"] = "NORMAL_ORDER"

    tier = source.get("selection_tier", pd.Series("STRICT", index=source.index)).astype(str)
    strict_mask = tier.eq("STRICT")
    backfill_mask = tier.eq("BACKFILL")

    source.loc[strict_mask, "execution_action"] = "BET"
    source.loc[backfill_mask, "execution_action"] = "OBSERVE_ONLY"
    source.loc[backfill_mask, "order_intent"] = "NO_ORDER_BACKFILL_SHADOW"
    source.loc[backfill_mask, "stake_units"] = 0.0
    source.loc[backfill_mask, "profit_units"] = 0.0

    recommendations = source.sort_values(
        ["game_date", "execution_action", "selection_tier", "rank_score", "game_pk"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    executable = recommendations[recommendations["execution_action"].eq("BET")].copy().reset_index(drop=True)
    observe_only = recommendations[recommendations["execution_action"].eq("OBSERVE_ONLY")].copy().reset_index(drop=True)

    outputs[f"{prefix}_orders"] = executable
    outputs[f"{prefix}_observe_only"] = observe_only
    outputs[f"{prefix}_recommendations"] = recommendations
    outputs[f"{prefix}_overall"] = pd.DataFrame([_summary_row_v30(executable, "STRICT_EXECUTABLE", result_col)])
    outputs[f"{prefix}_tier_summary"] = _strict_tier_summary(recommendations, market=market, result_col=result_col)
    bet_snapshots, closing_snapshots = build_snapshot_tables(executable)
    outputs[f"{prefix}_bet_snapshots"] = bet_snapshots
    outputs[f"{prefix}_closing_snapshots"] = closing_snapshots


def _shadow_summary_row(df: pd.DataFrame, label: str, result_col: str) -> dict:
    row = _summary_row_v30(df, label, result_col)
    shadow_stake = pd.to_numeric(df.get("shadow_stake_units", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    shadow_profit = pd.to_numeric(df.get("shadow_profit_units", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    row["shadow_stake_units"] = float(shadow_stake.sum()) if len(df) else 0.0
    row["shadow_profit_units"] = float(shadow_profit.sum()) if len(df) else 0.0
    row["shadow_roi"] = row["shadow_profit_units"] / row["shadow_stake_units"] if row["shadow_stake_units"] else 0.0
    return row


def _strict_tier_summary(df: pd.DataFrame, *, market: str, result_col: str) -> pd.DataFrame:
    rows = []
    strict = df[df.get("selection_tier", pd.Series(dtype=str)).astype(str).eq("STRICT")] if not df.empty else _empty_like(df)
    backfill = df[df.get("selection_tier", pd.Series(dtype=str)).astype(str).eq("BACKFILL")] if not df.empty else _empty_like(df)
    rows.append({"market": market, "tier_policy": "EXECUTABLE", **_shadow_summary_row(strict, "STRICT", result_col)})
    rows.append({"market": market, "tier_policy": "OBSERVE_ONLY", **_shadow_summary_row(backfill, "BACKFILL", result_col)})
    return pd.DataFrame(rows)


def run_strict_rc1(
    *,
    ml_scored_candidates: pd.DataFrame,
    ou_predictions: pd.DataFrame | None,
    features: pd.DataFrame | None,
    cfg: MarketIntelligenceV30StrictRC1Config,
) -> dict[str, pd.DataFrame]:
    outputs = run_v30(
        ml_scored_candidates=ml_scored_candidates,
        ou_predictions=ou_predictions,
        features=features,
        cfg=cfg,
    )
    _apply_strict_freeze(outputs, market="ML", prefix="ml", result_col="actual_result")
    if ou_predictions is not None:
        _apply_strict_freeze(outputs, market="OU", prefix="ou", result_col="game_result")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen MLB v3.0 strict-only RC1. Backfill is shadow/observe-only.")
    parser.add_argument("--ml-scored-candidates", required=True, type=Path)
    parser.add_argument("--ou-predictions", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-start-season", default=MarketIntelligenceV30StrictRC1Config.train_start_season, type=int)
    parser.add_argument("--train-end-season", default=MarketIntelligenceV30StrictRC1Config.train_end_season, type=int)
    parser.add_argument("--test-season", default=MarketIntelligenceV30StrictRC1Config.test_season, type=int)
    parser.add_argument("--snapshot-mode", choices=["historical_backtest", "live_paper"], default=MarketIntelligenceV30StrictRC1Config.snapshot_mode)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MarketIntelligenceV30StrictRC1Config(
        train_start_season=args.train_start_season,
        train_end_season=args.train_end_season,
        test_season=args.test_season,
        snapshot_mode=args.snapshot_mode,
    )
    ml_scored = pd.read_csv(args.ml_scored_candidates, low_memory=False)
    ou_predictions = pd.read_csv(args.ou_predictions, low_memory=False) if args.ou_predictions else None
    features = pd.read_csv(args.features, low_memory=False) if args.features else None
    outputs = run_strict_rc1(
        ml_scored_candidates=ml_scored,
        ou_predictions=ou_predictions,
        features=features,
        cfg=cfg,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "frozen_version": FROZEN_VERSION,
            "generated_at_utc": utc_now_iso(),
            "ml_scored_hash": canonical_hash(ml_scored),
            "ou_predictions_hash": canonical_hash(ou_predictions) if ou_predictions is not None else None,
            "features_hash": canonical_hash(features) if features is not None else None,
            "config": asdict(cfg),
            "outputs": {name: canonical_hash(frame) for name, frame in outputs.items()},
        },
    )
    print(outputs["ml_overall"].to_string(index=False))
    if "ou_overall" in outputs:
        print(outputs["ou_overall"].to_string(index=False))


if __name__ == "__main__":
    main()
