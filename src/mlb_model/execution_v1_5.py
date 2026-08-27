from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from .execution_v1_2 import (
    MLBExecutionV12Config,
    _append_reason,
    _base_predictions,
    _grade,
    _select_orders,
    _winner_candidates,
)
from .governance import canonical_hash, utc_now_iso, write_manifest
from .ou_calibration import FEATURE_COLUMNS_CATEGORICAL, FEATURE_COLUMNS_NUMERIC
from .synthetic_totals import add_synthetic_total_line


EXECUTION_VERSION = "mlb_execution_v1_5"
EXPECTED_CALIBRATION_VERSION = "mlb_ou_calibration_v1_3"


@dataclass(frozen=True)
class MLBExecutionV15Config(MLBExecutionV12Config):
    min_calibrated_ou_probability: float = 0.525
    min_calibrated_ou_ev: float = 0.0
    max_daily_ou_picks: int = 4
    no_bet_edge_band: float = 0.25
    max_vig: float = 0.08
    synthetic_divergence_penalty: float = 0.15
    odds_format: Literal["american", "decimal"] = "american"
    execution_version: str = EXECUTION_VERSION


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_metadata(model_path: Path) -> dict:
    metadata_path = model_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise ValueError(f"Missing calibrator metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _assert_schema(model_input: pd.DataFrame, metadata: dict) -> None:
    expected = metadata.get("feature_schema", {}).get("all")
    if not expected:
        raise ValueError("Calibrator metadata missing feature_schema.all")
    if list(model_input.columns) != list(expected):
        raise ValueError(f"Calibration feature schema mismatch. expected={expected}, actual={list(model_input.columns)}")
    if metadata.get("calibration_version") != EXPECTED_CALIBRATION_VERSION:
        raise ValueError(f"Unexpected calibration version: {metadata.get('calibration_version')}")


def _profit_per_unit(price: pd.Series, odds_format: Literal["american", "decimal"]) -> pd.Series:
    odds = pd.to_numeric(price, errors="coerce")
    profit = pd.Series(np.nan, index=price.index, dtype=float)
    if odds_format == "decimal":
        valid = odds.between(1.01, 20.0)
        profit.loc[valid] = odds.loc[valid] - 1.0
    else:
        valid = odds.abs() >= 80
        pos = valid & (odds > 0)
        neg = valid & (odds < 0)
        profit.loc[pos] = odds.loc[pos] / 100.0
        profit.loc[neg] = 100.0 / odds.loc[neg].abs()
    return profit


def _implied_probability(price: pd.Series, odds_format: Literal["american", "decimal"]) -> pd.Series:
    odds = pd.to_numeric(price, errors="coerce")
    implied = pd.Series(np.nan, index=price.index, dtype=float)
    if odds_format == "decimal":
        valid = odds.between(1.01, 20.0)
        implied.loc[valid] = 1.0 / odds.loc[valid]
    else:
        pos = odds >= 80
        neg = odds <= -80
        implied.loc[pos] = 100.0 / (odds.loc[pos] + 100.0)
        implied.loc[neg] = odds.loc[neg].abs() / (odds.loc[neg].abs() + 100.0)
    return implied


def _default_price(odds_format: Literal["american", "decimal"]) -> float:
    return -110.0 if odds_format == "american" else 1.9091


def _price_for_wagers(wagers: pd.DataFrame, odds_format: Literal["american", "decimal"]) -> pd.Series:
    price = pd.Series(_default_price(odds_format), index=wagers.index, dtype=float)
    if "market" not in wagers.columns:
        return price
    winner = wagers["market"].eq("WINNER")
    ou = wagers["market"].eq("OU")
    if "home_moneyline" in wagers.columns:
        mask = winner & wagers["side"].eq("HOME")
        price.loc[mask] = pd.to_numeric(wagers.loc[mask, "home_moneyline"], errors="coerce").fillna(price.loc[mask])
    if "away_moneyline" in wagers.columns:
        mask = winner & wagers["side"].eq("AWAY")
        price.loc[mask] = pd.to_numeric(wagers.loc[mask, "away_moneyline"], errors="coerce").fillna(price.loc[mask])
    if "total_price_over" in wagers.columns:
        mask = ou & wagers["side"].eq("OVER")
        price.loc[mask] = pd.to_numeric(wagers.loc[mask, "total_price_over"], errors="coerce").fillna(price.loc[mask])
    if "total_price_under" in wagers.columns:
        mask = ou & wagers["side"].eq("UNDER")
        price.loc[mask] = pd.to_numeric(wagers.loc[mask, "total_price_under"], errors="coerce").fillna(price.loc[mask])
    return price


def _summarize_v1_5(frame: pd.DataFrame, period: Literal["overall", "daily", "weekly"], odds_format: Literal["american", "decimal"]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    group_cols = ["execution_mode", "market"]
    if period == "daily":
        group_cols.append("game_date")
    if period == "weekly":
        dates = pd.to_datetime(out["game_date"], errors="raise")
        iso = dates.dt.isocalendar()
        out["test_week"] = iso["week"].astype(int)
        out["week_start"] = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.date.astype(str)
        group_cols.extend(["test_week", "week_start"])

    rows = []
    for keys, group in out.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        wagers = group[group["execution_action"].isin(["BET", "THROTTLE"])]
        row.update(
            {
                "games": int(len(group)),
                "orders": int(len(wagers)),
                "observe_only": int(group["execution_action"].eq("OBSERVE_ONLY").sum()),
                "blocked": int(group["execution_action"].eq("BLOCK").sum()),
                "stake_units": float(wagers["stake_final"].sum()) if "stake_final" in wagers else 0.0,
                "avg_edge_abs": float(wagers["edge_abs"].mean()) if len(wagers) else 0.0,
                "avg_probability": float(wagers["heuristic_probability_score"].mean()) if len(wagers) else 0.0,
                "wins": int(wagers["actual_result"].eq("WIN").sum()) if "actual_result" in wagers else 0,
                "losses": int(wagers["actual_result"].eq("LOSS").sum()) if "actual_result" in wagers else 0,
                "pushes": int(wagers["actual_result"].eq("PUSH").sum()) if "actual_result" in wagers else 0,
            }
        )
        row["profit_units"] = 0.0
        if "actual_result" in wagers:
            prices = _price_for_wagers(wagers, odds_format)
            per_unit = _profit_per_unit(prices, odds_format).fillna(0.0)
            row["profit_units"] = float(
                (wagers["actual_result"].eq("WIN") * wagers["stake_final"] * per_unit).sum()
                - (wagers["actual_result"].eq("LOSS") * wagers["stake_final"]).sum()
            )
        decisions = row["wins"] + row["losses"]
        row["win_rate"] = float(row["wins"] / decisions) if decisions else 0.0
        row["roi"] = float(row["profit_units"] / row["stake_units"]) if row["stake_units"] else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _calibrated_ou_candidates(predictions: pd.DataFrame, cfg: MLBExecutionV15Config, calibrator, metadata: dict) -> pd.DataFrame:
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
    out["side"] = np.where(out["model_minus_market"] > cfg.no_bet_edge_band, "OVER", "UNDER")
    out.loc[out["model_minus_market"].abs() <= cfg.no_bet_edge_band, "side"] = "NO_BET_BAND"
    out["display_side"] = out["side"]
    out["edge"] = np.where(out["side"].eq("OVER"), out["model_minus_market"], -out["model_minus_market"]).round(3)
    out["edge_abs"] = out["edge"].abs()
    out["price"] = np.where(
        out["side"].eq("OVER"),
        pd.to_numeric(out.get("total_price_over", pd.NA), errors="coerce"),
        pd.to_numeric(out.get("total_price_under", pd.NA), errors="coerce"),
    )
    out["payout_per_unit"] = _profit_per_unit(pd.Series(out["price"], index=out.index), cfg.odds_format)
    out["implied_probability"] = _implied_probability(pd.Series(out["price"], index=out.index), cfg.odds_format)
    over_imp = _implied_probability(pd.to_numeric(out.get("total_price_over", pd.NA), errors="coerce"), cfg.odds_format)
    under_imp = _implied_probability(pd.to_numeric(out.get("total_price_under", pd.NA), errors="coerce"), cfg.odds_format)
    out["market_vig"] = over_imp + under_imp - 1.0

    out["execution_action"] = "ELIGIBLE"
    out["block_reason"] = ""
    missing_line = out["market_total"].isna()
    out.loc[missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], missing_line, "MISSING_TOTAL_LINE")

    model_input = out[FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL].copy()
    _assert_schema(model_input, metadata)
    probabilities = pd.Series(np.nan, index=out.index, dtype=float)
    valid_side = out["side"].isin(["OVER", "UNDER"])
    if valid_side.any():
        probabilities.loc[valid_side] = calibrator.predict_proba(model_input.loc[valid_side])[:, 1]
    out["calibrated_probability"] = probabilities
    out["heuristic_probability_score"] = out["calibrated_probability"]
    out["calibrated_ev_raw"] = out["calibrated_probability"] * out["payout_per_unit"] - (1.0 - out["calibrated_probability"])
    divergence = out["abs_market_minus_synthetic"].fillna(0.0)
    out["synthetic_divergence_penalty"] = (divergence / cfg.max_synthetic_market_gap).clip(lower=0, upper=1) * cfg.synthetic_divergence_penalty
    out["calibrated_ev"] = out["calibrated_ev_raw"] - out["synthetic_divergence_penalty"]
    out["selection_score"] = (
        0.70 * out["calibrated_ev"].fillna(-1)
        + 0.20 * out["calibrated_probability"].fillna(0)
        + 0.10 * (out["edge_abs"].fillna(0) / cfg.max_ou_edge).clip(0, 1)
    ).round(6)

    checks = [
        (out["calibrated_probability"].isna(), "MISSING_CALIBRATION"),
        (out["implied_probability"].isna() | ~out["implied_probability"].between(0.01, 0.99), "INVALID_PRICE"),
        (out["market_vig"].notna() & (out["market_vig"] > cfg.max_vig), "HIGH_VIG"),
        (out["side"].eq("NO_BET_BAND"), "NO_BET_EDGE_BAND"),
        ((out["edge_abs"] < cfg.min_ou_edge) & ~missing_line, "MIN_OU_EDGE"),
        ((out["edge_abs"] > cfg.max_ou_edge) & ~missing_line, "MAX_OU_EDGE_MODEL_BUG_GUARDRAIL"),
        ((out["calibrated_probability"] < cfg.min_calibrated_ou_probability) & ~missing_line, "MIN_CALIBRATED_PROB"),
        ((out["calibrated_ev"] < cfg.min_calibrated_ou_ev) & ~missing_line, "MIN_CALIBRATED_EV"),
    ]
    for mask, reason in checks:
        out.loc[mask, "execution_action"] = "BLOCK"
        out["block_reason"] = _append_reason(out["block_reason"], mask, reason)
    out["extreme_edge_audit"] = np.where((out["edge_abs"] > cfg.max_ou_edge) & ~missing_line, "REVIEW_MODEL_OR_LINE", "")
    return out


def execute_v1_5(
    predictions: pd.DataFrame,
    *,
    ou_calibrator_path: Path,
    config: MLBExecutionV15Config | None = None,
    capital_state: dict | None = None,
    risk_state: dict | None = None,
    previous_action: str | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or MLBExecutionV15Config()
    base = _base_predictions(predictions)
    calibrator = joblib.load(ou_calibrator_path)
    metadata = _load_metadata(ou_calibrator_path)
    parts = []
    if cfg.winner_enabled:
        parts.append(_winner_candidates(base, cfg))
    if cfg.market_ou_enabled:
        parts.append(_calibrated_ou_candidates(base, cfg, calibrator, metadata))
    candidates = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    candidates = _grade(candidates)
    candidates["execution_version"] = EXECUTION_VERSION
    candidates["input_hash"] = canonical_hash(predictions)
    candidates["config_hash"] = canonical_hash(asdict(cfg))
    candidates["ou_calibrator_training_hash"] = metadata.get("training_hash")
    candidates["ou_calibrator_file_hash"] = _file_hash(ou_calibrator_path)
    candidates["ou_calibrator_metadata_hash"] = canonical_hash(metadata)
    selected = _select_orders(
        candidates,
        cfg,
        capital_state=capital_state,
        risk_state=risk_state,
        previous_action=previous_action,
    )
    selected["action_sort"] = selected["execution_action"].map({"BET": 0, "THROTTLE": 1, "OBSERVE_ONLY": 2, "BLOCK": 3}).fillna(9)
    selected["generated_at_utc"] = utc_now_iso()
    selected = selected.sort_values(
        ["game_date", "execution_mode", "action_sort", "selection_score", "game_pk", "side"],
        ascending=[True, True, True, False, True, True],
        kind="mergesort",
    ).drop(columns=["action_sort"]).reset_index(drop=True)
    selected["execution_hash"] = canonical_hash(selected.drop(columns=["generated_at_utc", "execution_hash"], errors="ignore"))
    orders = selected[selected["execution_action"].isin(["BET", "THROTTLE"])].copy().reset_index(drop=True)
    return {
        "orders": orders,
        "audit_candidates": selected,
        "daily_summary": _summarize_v1_5(selected, "daily", cfg.odds_format),
        "weekly_summary": _summarize_v1_5(selected, "weekly", cfg.odds_format),
        "overall_summary": _summarize_v1_5(selected, "overall", cfg.odds_format),
    }


def _load_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLB execution v1.5 with explicit odds format and certified calibrator.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ou-calibrator", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--odds-format", choices=["american", "decimal"], default=MLBExecutionV15Config.odds_format)
    parser.add_argument("--capital-state", type=Path)
    parser.add_argument("--risk-state", type=Path)
    parser.add_argument("--previous-action")
    parser.add_argument("--winner-daily-cap", default=MLBExecutionV15Config.max_daily_winner_picks, type=int)
    parser.add_argument("--ou-daily-cap", default=MLBExecutionV15Config.max_daily_ou_picks, type=int)
    parser.add_argument("--min-ou-probability", default=MLBExecutionV15Config.min_calibrated_ou_probability, type=float)
    parser.add_argument("--min-ou-ev", default=MLBExecutionV15Config.min_calibrated_ou_ev, type=float)
    parser.add_argument("--max-vig", default=MLBExecutionV15Config.max_vig, type=float)
    parser.add_argument("--no-bet-edge-band", default=MLBExecutionV15Config.no_bet_edge_band, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    cfg = MLBExecutionV15Config(
        max_daily_winner_picks=args.winner_daily_cap,
        max_daily_ou_picks=args.ou_daily_cap,
        min_calibrated_ou_probability=args.min_ou_probability,
        min_calibrated_ou_ev=args.min_ou_ev,
        max_vig=args.max_vig,
        no_bet_edge_band=args.no_bet_edge_band,
        odds_format=args.odds_format,
    )
    outputs = execute_v1_5(
        predictions,
        ou_calibrator_path=args.ou_calibrator,
        config=cfg,
        capital_state=_load_json(args.capital_state),
        risk_state=_load_json(args.risk_state),
        previous_action=args.previous_action,
    )
    metadata = _load_metadata(args.ou_calibrator)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "execution_version": EXECUTION_VERSION,
            "generated_at_utc": utc_now_iso(),
            "predictions_hash": canonical_hash(predictions),
            "ou_calibrator": str(args.ou_calibrator),
            "ou_calibrator_file_hash": _file_hash(args.ou_calibrator),
            "ou_calibrator_metadata_hash": canonical_hash(metadata),
            "config": asdict(cfg),
            "outputs": {name: canonical_hash(frame) for name, frame in outputs.items()},
        },
    )
    print(outputs["overall_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
