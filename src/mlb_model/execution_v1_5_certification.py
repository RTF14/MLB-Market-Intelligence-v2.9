from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .execution_v1_5 import (
    EXECUTION_VERSION,
    MLBExecutionV15Config,
    _calibrated_ou_candidates,
    _file_hash,
    _load_metadata,
    execute_v1_5,
)
from .governance import canonical_hash, utc_now_iso, write_manifest
from .ou_calibration import FEATURE_COLUMNS_CATEGORICAL, FEATURE_COLUMNS_NUMERIC


class ConstantCalibrator:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, model_input: pd.DataFrame) -> np.ndarray:
        probability = np.full(len(model_input), self.probability, dtype=float)
        return np.column_stack([1.0 - probability, probability])


def _metadata() -> dict:
    return {
        "calibration_version": "mlb_ou_calibration_v1_3",
        "feature_schema": {
            "numeric": FEATURE_COLUMNS_NUMERIC,
            "categorical": FEATURE_COLUMNS_CATEGORICAL,
            "all": FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL,
        },
        "training_hash": "certification-fixture",
    }


def _fixture(**overrides) -> pd.DataFrame:
    row = {
        "season": 2025,
        "game_date": "2025-04-01",
        "game_pk": 900001,
        "home_team": "Home",
        "away_team": "Away",
        "home_score": 6,
        "away_score": 4,
        "total_runs": 10,
        "pred_home_score": 5.4,
        "pred_away_score": 4.6,
        "pred_total": 10.0,
        "pred_margin": 0.8,
        "total_line": 8.0,
        "synthetic_total_line": 8.0,
        "total_price_over": -110.0,
        "total_price_under": -110.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _assert_reason(frame: pd.DataFrame, reason: str) -> None:
    reasons = frame["block_reason"].fillna("").astype(str)
    if not reasons.str.contains(reason, regex=False).any():
        raise AssertionError(f"Expected block reason {reason}, got {reasons.tolist()}")


def _expect_raises(name: str, fn, expected: str) -> dict:
    try:
        fn()
    except Exception as exc:
        if expected not in str(exc):
            raise AssertionError(f"{name} raised wrong error: {exc}") from exc
        return {"test": name, "status": "PASS", "detail": str(exc)}
    raise AssertionError(f"{name} did not raise")


def _run_gate_tests() -> list[dict]:
    cfg = MLBExecutionV15Config(market_ou_enabled=True, winner_enabled=False)
    metadata = _metadata()
    tests: list[dict] = []

    cases = [
        (
            "missing probability blocks order",
            _fixture(),
            ConstantCalibrator(np.nan),
            "MISSING_CALIBRATION",
        ),
        (
            "invalid price blocks order",
            _fixture(total_price_over=50.0),
            ConstantCalibrator(0.70),
            "INVALID_PRICE",
        ),
        (
            "high vig blocks order",
            _fixture(total_price_over=-200.0, total_price_under=-200.0),
            ConstantCalibrator(0.70),
            "HIGH_VIG",
        ),
        (
            "no-bet band blocks order",
            _fixture(pred_total=8.10),
            ConstantCalibrator(0.70),
            "NO_BET_EDGE_BAND",
        ),
    ]
    for name, fixture, calibrator, reason in cases:
        result = _calibrated_ou_candidates(fixture, cfg, calibrator, metadata)
        _assert_reason(result, reason)
        if result["execution_action"].isin(["BET", "THROTTLE", "ELIGIBLE"]).any():
            raise AssertionError(f"{name} left an executable row")
        tests.append({"test": name, "status": "PASS", "detail": reason})

    divergence = _calibrated_ou_candidates(
        _fixture(synthetic_total_line=5.0),
        cfg,
        ConstantCalibrator(0.70),
        metadata,
    )
    raw_ev = float(divergence.loc[0, "calibrated_ev_raw"])
    penalized_ev = float(divergence.loc[0, "calibrated_ev"])
    if not penalized_ev < raw_ev:
        raise AssertionError("Synthetic divergence did not penalize EV")
    if "SYNTHETIC" in str(divergence.loc[0, "block_reason"]):
        raise AssertionError("Synthetic divergence hard-blocked the row")
    tests.append(
        {
            "test": "synthetic divergence penalizes EV but does not hard block",
            "status": "PASS",
            "detail": f"raw_ev={raw_ev:.6f}; penalized_ev={penalized_ev:.6f}",
        }
    )
    return tests


def _run_model_contract_tests(predictions: pd.DataFrame, model_path: Path, out_dir: Path) -> list[dict]:
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model_copy = tmp_dir / "ou_calibrator_no_metadata.joblib"
    shutil.copy2(model_path, model_copy)
    tests = [
        _expect_raises(
            "missing calibrator metadata blocks run",
            lambda: execute_v1_5(
                predictions.head(5),
                ou_calibrator_path=model_copy,
                config=MLBExecutionV15Config(winner_enabled=False),
            ),
            "Missing calibrator metadata",
        )
    ]

    bad_model = tmp_dir / "ou_calibrator_bad_schema.joblib"
    shutil.copy2(model_path, bad_model)
    bad_metadata = _load_metadata(model_path).copy()
    bad_metadata["feature_schema"] = {
        "numeric": ["model_total"],
        "categorical": [],
        "all": ["model_total"],
    }
    bad_model.with_suffix(".metadata.json").write_text(json.dumps(bad_metadata, sort_keys=True), encoding="utf-8")
    tests.append(
        _expect_raises(
            "feature schema mismatch blocks run",
            lambda: execute_v1_5(
                predictions.head(5),
                ou_calibrator_path=bad_model,
                config=MLBExecutionV15Config(winner_enabled=False),
            ),
            "Calibration feature schema mismatch",
        )
    )
    return tests


def _run_replay_and_cap_tests(predictions: pd.DataFrame, model_path: Path) -> list[dict]:
    cfg = MLBExecutionV15Config(max_daily_ou_picks=2)
    first = execute_v1_5(predictions, ou_calibrator_path=model_path, config=cfg)
    second = execute_v1_5(predictions, ou_calibrator_path=model_path, config=cfg)
    first_hashes = sorted(first["audit_candidates"]["execution_hash"].dropna().unique().tolist())
    second_hashes = sorted(second["audit_candidates"]["execution_hash"].dropna().unique().tolist())
    if first_hashes != second_hashes:
        raise AssertionError("Replay hash changed between identical runs")

    ou_orders = first["orders"][first["orders"]["execution_mode"].eq("market_ou_calibrated")]
    daily_counts = ou_orders.groupby("game_date").size()
    max_daily = int(daily_counts.max()) if not daily_counts.empty else 0
    if max_daily > cfg.max_daily_ou_picks:
        raise AssertionError(f"Max daily OU cap violated: {max_daily} > {cfg.max_daily_ou_picks}")
    return [
        {
            "test": "replay hash stable excluding timestamp",
            "status": "PASS",
            "detail": first_hashes[0] if first_hashes else "no-hash",
        },
        {
            "test": "max daily OU cap enforced",
            "status": "PASS",
            "detail": f"max_daily_ou_orders={max_daily}",
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify MLB execution v1.5 governance gates.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ou-calibrator", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    results.extend(_run_gate_tests())
    results.extend(_run_model_contract_tests(predictions, args.ou_calibrator, args.out_dir))
    results.extend(_run_replay_and_cap_tests(predictions, args.ou_calibrator))

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(args.out_dir / "certification_results.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "execution_version": EXECUTION_VERSION,
            "generated_at_utc": utc_now_iso(),
            "predictions_hash": canonical_hash(predictions),
            "ou_calibrator": str(args.ou_calibrator),
            "ou_calibrator_file_hash": _file_hash(args.ou_calibrator),
            "results_hash": canonical_hash(results_frame),
            "tests": results,
        },
    )
    print(results_frame.to_string(index=False))


if __name__ == "__main__":
    main()
