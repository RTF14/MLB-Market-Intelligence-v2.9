from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .execution_v1_2 import (
    MLBExecutionV12Config,
    _append_reason,
    _base_predictions,
    _grade,
    _select_orders,
    _summarize,
    _winner_candidates,
)
from .governance import canonical_hash, utc_now_iso, write_manifest
from .ou_calibration import (
    FEATURE_COLUMNS_CATEGORICAL,
    FEATURE_COLUMNS_NUMERIC,
    _american_profit_per_unit,
)
from .synthetic_totals import add_synthetic_total_line


EXECUTION_VERSION = "mlb_execution_v1_3"


@dataclass(frozen=True)
class MLBExecutionV13Config(MLBExecutionV12Config):
    min_calibrated_ou_probability: float = 0.525
    min_calibrated_ou_ev: float = 0.0
    max_daily_ou_picks: int = 4
    execution_version: str = EXECUTION_VERSION


def _calibrated_ou_candidates(predictions: pd.DataFrame, cfg: MLBExecutionV13Config, calibrator) -> pd.DataFrame:
    out = predictions.copy()
    out = add_synthetic_total_line(out) if "synthetic_total_line" not in out.columns else out
    out["market"] = "OU"
    out["execution_mode"] = "market_ou_calibrated"
    out["model_total"] = pd.to_numeric(out["pred_total"], errors="coerce")
    out["market_total"] = pd.to_numeric(out.get("total_line", pd.NA), errors="coerce")
    out["synthetic_total"] = pd.to_numeric(out["synthetic_total_line"], errors="coerce")
    out["model_minus_market"] = out["model_total"] - out["market_total"]
    out["model_minus_synthetic"] = out["model_total"] - out["synthetic_total"]
    out["market_minus_synthetic"] = out["market_total"] - out["synthetic_total"]
    out["abs_model_minus_market"] = out["model_minus_market"].abs()
    out["abs_market_minus_synthetic"] = out["market_minus_synthetic"].abs()
    out["closing_total_bucket"] = (out["market_total"] * 2).round() / 2
    out["month"] = pd.to_datetime(out["game_date"], errors="raise").dt.month.astype(int)
    out["side"] = np.where(out["model_minus_market"] > 0, "OVER", "UNDER")
    out["display_side"] = out["side"]
    out["edge"] = np.where(out["side"].eq("OVER"), out["model_minus_market"], -out["model_minus_market"]).round(3)
    out["edge_abs"] = out["edge"].abs()
    out["price"] = np.where(
        out["side"].eq("OVER"),
        pd.to_numeric(out.get("total_price_over", -110), errors="coerce"),
        pd.to_numeric(out.get("total_price_under", -110), errors="coerce"),
    )
    out["payout_per_unit"] = _american_profit_per_unit(pd.Series(out["price"], index=out.index))
    out["execution_action"] = "ELIGIBLE"
    out["block_reason"] = ""

    missing_line = out["market_total"].isna()
    out.loc[missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], missing_line, "MISSING_TOTAL_LINE")

    model_input = out[FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL].copy()
    valid_features = ~model_input[FEATURE_COLUMNS_NUMERIC].isna().all(axis=1)
    probabilities = pd.Series(np.nan, index=out.index, dtype=float)
    if valid_features.any():
        probabilities.loc[valid_features] = calibrator.predict_proba(model_input.loc[valid_features])[:, 1]
    out["calibrated_probability"] = probabilities
    out["heuristic_probability_score"] = out["calibrated_probability"]
    out["calibrated_ev"] = out["calibrated_probability"] * out["payout_per_unit"] - (1.0 - out["calibrated_probability"])
    out["selection_score"] = (
        out["calibrated_ev"] * 100.0 + out["calibrated_probability"] + out["edge_abs"] / 100.0
    ).round(6)

    low_edge = out["edge_abs"] < cfg.min_ou_edge
    high_edge = out["edge_abs"] > cfg.max_ou_edge
    low_prob = out["calibrated_probability"] < cfg.min_calibrated_ou_probability
    low_ev = out["calibrated_ev"] < cfg.min_calibrated_ou_ev
    synthetic_gap = out["abs_market_minus_synthetic"] > cfg.max_synthetic_market_gap
    out.loc[low_edge & ~missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], low_edge & ~missing_line, "MIN_OU_EDGE")
    out.loc[high_edge & ~missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], high_edge & ~missing_line, "MAX_OU_EDGE")
    out.loc[low_prob & ~missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], low_prob & ~missing_line, "MIN_CALIBRATED_PROB")
    out.loc[low_ev & ~missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], low_ev & ~missing_line, "MIN_CALIBRATED_EV")
    out.loc[synthetic_gap & ~missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], synthetic_gap & ~missing_line, "MARKET_SYNTHETIC_DIVERGENCE")
    return out


def execute_v1_3(
    predictions: pd.DataFrame,
    *,
    ou_calibrator_path: Path,
    config: MLBExecutionV13Config | None = None,
    capital_state: dict | None = None,
    risk_state: dict | None = None,
    previous_action: str | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or MLBExecutionV13Config()
    base = _base_predictions(predictions)
    calibrator = joblib.load(ou_calibrator_path)
    parts = []
    if cfg.winner_enabled:
        parts.append(_winner_candidates(base, cfg))
    if cfg.market_ou_enabled:
        parts.append(_calibrated_ou_candidates(base, cfg, calibrator))
    candidates = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    candidates = _grade(candidates)
    candidates["execution_version"] = EXECUTION_VERSION
    candidates["input_hash"] = canonical_hash(predictions)
    candidates["config_hash"] = canonical_hash(asdict(cfg))
    candidates["ou_calibrator_hash"] = canonical_hash({"path": str(ou_calibrator_path), "version": "mlb_ou_calibration_v1_3"})
    selected = _select_orders(
        candidates,
        cfg,
        capital_state=capital_state,
        risk_state=risk_state,
        previous_action=previous_action,
    )
    selected["generated_at_utc"] = utc_now_iso()
    selected = selected.sort_values(
        ["game_date", "execution_mode", "execution_action", "selection_score", "game_pk", "side"],
        ascending=[True, True, True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected["execution_hash"] = canonical_hash(selected.drop(columns=["generated_at_utc", "execution_hash"], errors="ignore"))
    orders = selected[selected["execution_action"].isin(["BET", "THROTTLE"])].copy().reset_index(drop=True)
    return {
        "orders": orders,
        "audit_candidates": selected,
        "daily_summary": _summarize(selected, "daily"),
        "weekly_summary": _summarize(selected, "weekly"),
        "overall_summary": _summarize(selected, "overall"),
    }


def _load_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLB execution v1.3 with calibrated O/U.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ou-calibrator", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--capital-state", type=Path)
    parser.add_argument("--risk-state", type=Path)
    parser.add_argument("--previous-action")
    parser.add_argument("--winner-daily-cap", default=MLBExecutionV13Config.max_daily_winner_picks, type=int)
    parser.add_argument("--ou-daily-cap", default=MLBExecutionV13Config.max_daily_ou_picks, type=int)
    parser.add_argument("--min-ou-probability", default=MLBExecutionV13Config.min_calibrated_ou_probability, type=float)
    parser.add_argument("--min-ou-ev", default=MLBExecutionV13Config.min_calibrated_ou_ev, type=float)
    parser.add_argument("--max-synthetic-market-gap", default=MLBExecutionV13Config.max_synthetic_market_gap, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    cfg = MLBExecutionV13Config(
        max_daily_winner_picks=args.winner_daily_cap,
        max_daily_ou_picks=args.ou_daily_cap,
        min_calibrated_ou_probability=args.min_ou_probability,
        min_calibrated_ou_ev=args.min_ou_ev,
        max_synthetic_market_gap=args.max_synthetic_market_gap,
    )
    outputs = execute_v1_3(
        predictions,
        ou_calibrator_path=args.ou_calibrator,
        config=cfg,
        capital_state=_load_json(args.capital_state),
        risk_state=_load_json(args.risk_state),
        previous_action=args.previous_action,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "execution_version": EXECUTION_VERSION,
            "generated_at_utc": utc_now_iso(),
            "predictions_hash": canonical_hash(predictions),
            "ou_calibrator": args.ou_calibrator,
            "config": asdict(cfg),
            "outputs": {name: canonical_hash(frame) for name, frame in outputs.items()},
        },
    )
    print(outputs["overall_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
