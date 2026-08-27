from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .governance import canonical_hash, utc_now_iso, write_manifest
from .synthetic_totals import add_synthetic_total_line


CALIBRATION_VERSION = "mlb_ou_calibration_v1_3"


@dataclass(frozen=True)
class OUCalibrationConfig:
    train_start_season: int = 2021
    train_end_season: int = 2024
    test_season: int = 2025
    min_abs_edge: float = 0.75
    max_abs_edge: float = 8.0
    min_ev: float = 0.0
    min_probability: float = 0.525
    max_daily_picks: int = 4
    calibration_version: str = CALIBRATION_VERSION


FEATURE_COLUMNS_NUMERIC = [
    "model_total",
    "market_total",
    "synthetic_total",
    "model_minus_market",
    "model_minus_synthetic",
    "market_minus_synthetic",
    "abs_model_minus_market",
    "abs_market_minus_synthetic",
    "closing_total_bucket",
    "month",
]

FEATURE_COLUMNS_CATEGORICAL = [
    "side",
]


def _american_profit_per_unit(price: pd.Series) -> pd.Series:
    odds = pd.to_numeric(price, errors="coerce")
    profit = pd.Series(0.9091, index=price.index, dtype=float)
    decimal = odds.between(1.01, 20.0)
    american = odds.abs() >= 80
    invalid = ~(decimal | american) | odds.isna()
    odds = odds.where(~invalid, -110.0)
    decimal = odds.between(1.01, 20.0)
    american_positive = (odds > 0) & ~decimal
    american_negative = (odds < 0) & ~decimal
    profit.loc[decimal] = odds.loc[decimal] - 1.0
    profit.loc[american_positive] = odds.loc[american_positive] / 100.0
    profit.loc[american_negative] = 100.0 / odds.loc[american_negative].abs()
    return profit


def build_ou_training_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "game_date", "game_pk", "pred_total", "total_line", "total_runs"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"O/U calibration input missing columns: {sorted(missing)}")

    out = predictions.copy()
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise")
    out["month"] = out["game_date"].dt.month.astype(int)
    out["model_total"] = pd.to_numeric(out["pred_total"], errors="coerce")
    out["market_total"] = pd.to_numeric(out["total_line"], errors="coerce")
    out["actual_total"] = pd.to_numeric(out["total_runs"], errors="coerce")
    if "synthetic_total_line" in out.columns:
        out["synthetic_total"] = pd.to_numeric(out["synthetic_total_line"], errors="coerce")
    else:
        out["synthetic_total"] = out["actual_total"].expanding().mean().shift(1)
        out["synthetic_total"] = out["synthetic_total"].fillna(out["actual_total"].mean())

    out = out.dropna(subset=["model_total", "market_total", "actual_total"]).copy()
    out["model_minus_market"] = out["model_total"] - out["market_total"]
    out["model_minus_synthetic"] = out["model_total"] - out["synthetic_total"]
    out["market_minus_synthetic"] = out["market_total"] - out["synthetic_total"]
    out["abs_model_minus_market"] = out["model_minus_market"].abs()
    out["abs_market_minus_synthetic"] = out["market_minus_synthetic"].abs()
    out["closing_total_bucket"] = (out["market_total"] * 2).round() / 2
    out["actual_over"] = out["actual_total"] > out["market_total"]
    out["actual_under"] = out["actual_total"] < out["market_total"]
    out["push"] = out["actual_total"] == out["market_total"]

    over = out.copy()
    over["side"] = "OVER"
    over["edge"] = over["model_minus_market"]
    over["abs_edge"] = over["edge"].abs()
    over["target_win"] = over["actual_over"]
    over["price"] = pd.to_numeric(over.get("total_price_over", -110), errors="coerce")

    under = out.copy()
    under["side"] = "UNDER"
    under["edge"] = -under["model_minus_market"]
    under["abs_edge"] = under["edge"].abs()
    under["target_win"] = under["actual_under"]
    under["price"] = pd.to_numeric(under.get("total_price_under", -110), errors="coerce")

    stacked = pd.concat([over, under], ignore_index=True, sort=False)
    stacked = stacked[~stacked["push"]].copy()
    stacked = stacked[stacked["edge"] > 0].copy()
    stacked["payout_per_unit"] = _american_profit_per_unit(stacked["price"])
    return stacked.sort_values(["game_date", "game_pk", "side"], kind="mergesort").reset_index(drop=True)


def build_pipeline() -> Pipeline:
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, FEATURE_COLUMNS_NUMERIC),
            ("categorical", categorical, FEATURE_COLUMNS_CATEGORICAL),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def fit_calibrator(frame: pd.DataFrame, cfg: OUCalibrationConfig, model_path: Path) -> tuple[Pipeline, pd.DataFrame]:
    train = frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)].copy()
    if train.empty:
        raise ValueError("No calibration training rows after filtering")
    model = build_pipeline()
    model.fit(train[FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL], train["target_win"].astype(int))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {
        "calibration_version": cfg.calibration_version,
        "model_type": "sklearn.pipeline.Pipeline(LogisticRegression)",
        "sklearn_version": sklearn.__version__,
        "training_hash": canonical_hash(train),
        "feature_schema": {
            "numeric": FEATURE_COLUMNS_NUMERIC,
            "categorical": FEATURE_COLUMNS_CATEGORICAL,
            "all": FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL,
        },
        "config": asdict(cfg),
        "generated_at_utc": utc_now_iso(),
    }
    write_manifest(model_path.with_suffix(".metadata.json"), metadata)
    return model, train


def score_calibrator(model: Pipeline, frame: pd.DataFrame, cfg: OUCalibrationConfig) -> pd.DataFrame:
    out = frame.copy()
    proba = model.predict_proba(out[FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL])[:, 1]
    out["calibrated_probability"] = proba
    out["calibrated_ev"] = out["calibrated_probability"] * out["payout_per_unit"] - (1.0 - out["calibrated_probability"])
    out["eligible"] = (
        (out["abs_edge"] >= cfg.min_abs_edge)
        & (out["abs_edge"] <= cfg.max_abs_edge)
        & (out["calibrated_probability"] >= cfg.min_probability)
        & (out["calibrated_ev"] >= cfg.min_ev)
    )
    out["rank_score"] = (out["calibrated_ev"] * 100.0 + out["calibrated_probability"] + out["abs_edge"] / 100.0).round(6)
    out["execution_action"] = "BLOCK"
    out["block_reason"] = "CALIBRATION_GATE"
    selected = []
    for (_season, game_date), group in out[out["eligible"]].groupby(["season", "game_date"], sort=True):
        ordered = group.sort_values(
            ["rank_score", "calibrated_ev", "calibrated_probability", "abs_edge", "game_pk", "side"],
            ascending=[False, False, False, False, True, True],
            kind="mergesort",
        ).head(cfg.max_daily_picks)
        selected.append(ordered)
    if selected:
        selected_idx = pd.concat(selected).index
        out.loc[selected_idx, "execution_action"] = "BET"
        out.loc[selected_idx, "block_reason"] = ""
    out["profit_units"] = 0.0
    out.loc[out["execution_action"].eq("BET") & out["target_win"], "profit_units"] = out["payout_per_unit"]
    out.loc[out["execution_action"].eq("BET") & ~out["target_win"], "profit_units"] = -1.0
    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise").dt.date.astype(str)
    return out.sort_values(["game_date", "execution_action", "rank_score"], ascending=[True, True, False]).reset_index(drop=True)


def summarize(scored: pd.DataFrame, cfg: OUCalibrationConfig) -> dict[str, pd.DataFrame]:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    bets = test[test["execution_action"].eq("BET")]
    overall = pd.DataFrame(
        [
            {
                "season": cfg.test_season,
                "candidates": int(len(test)),
                "bets": int(len(bets)),
                "wins": int(bets["target_win"].sum()) if len(bets) else 0,
                "losses": int((~bets["target_win"]).sum()) if len(bets) else 0,
                "win_rate": float(bets["target_win"].mean()) if len(bets) else 0.0,
                "profit_units": float(bets["profit_units"].sum()) if len(bets) else 0.0,
                "roi": float(bets["profit_units"].sum() / len(bets)) if len(bets) else 0.0,
                "avg_probability": float(bets["calibrated_probability"].mean()) if len(bets) else 0.0,
                "avg_ev": float(bets["calibrated_ev"].mean()) if len(bets) else 0.0,
            }
        ]
    )
    weekly_rows = []
    if not test.empty:
        dates = pd.to_datetime(test["game_date"], errors="raise")
        test["test_week"] = dates.dt.isocalendar()["week"].astype(int)
        test["week_start"] = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.date.astype(str)
        for (week, week_start), group in test.groupby(["test_week", "week_start"], sort=True):
            group_bets = group[group["execution_action"].eq("BET")]
            weekly_rows.append(
                {
                    "test_week": int(week),
                    "week_start": week_start,
                    "candidates": int(len(group)),
                    "bets": int(len(group_bets)),
                    "wins": int(group_bets["target_win"].sum()) if len(group_bets) else 0,
                    "losses": int((~group_bets["target_win"]).sum()) if len(group_bets) else 0,
                    "win_rate": float(group_bets["target_win"].mean()) if len(group_bets) else 0.0,
                    "profit_units": float(group_bets["profit_units"].sum()) if len(group_bets) else 0.0,
                    "roi": float(group_bets["profit_units"].sum() / len(group_bets)) if len(group_bets) else 0.0,
                }
            )
    return {"overall": overall, "weekly": pd.DataFrame(weekly_rows), "scored_candidates": scored}


def model_metrics(scored: pd.DataFrame, cfg: OUCalibrationConfig) -> pd.DataFrame:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    if test.empty:
        return pd.DataFrame()
    y = test["target_win"].astype(int)
    p = test["calibrated_probability"]
    metrics = {
        "season": cfg.test_season,
        "rows": int(len(test)),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "accuracy_at_50": accuracy_score(y, p >= 0.5),
    }
    if y.nunique() > 1:
        metrics["auc"] = roc_auc_score(y, p)
    return pd.DataFrame([metrics])


def run_calibration(predictions: pd.DataFrame, cfg: OUCalibrationConfig, model_path: Path) -> dict[str, pd.DataFrame]:
    if "synthetic_total_line" not in predictions.columns:
        predictions = add_synthetic_total_line(predictions)
    frame = build_ou_training_frame(predictions)
    model, train = fit_calibrator(frame, cfg, model_path)
    scored = score_calibrator(model, frame, cfg)
    result = summarize(scored, cfg)
    result["model_metrics"] = model_metrics(scored, cfg)
    result["training_rows"] = pd.DataFrame([{"rows": len(train), "hash": canonical_hash(train)}])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train O/U calibration on SBR lines and test a target season.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--train-start-season", default=OUCalibrationConfig.train_start_season, type=int)
    parser.add_argument("--train-end-season", default=OUCalibrationConfig.train_end_season, type=int)
    parser.add_argument("--test-season", default=OUCalibrationConfig.test_season, type=int)
    parser.add_argument("--min-ev", default=OUCalibrationConfig.min_ev, type=float)
    parser.add_argument("--min-probability", default=OUCalibrationConfig.min_probability, type=float)
    parser.add_argument("--max-daily-picks", default=OUCalibrationConfig.max_daily_picks, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OUCalibrationConfig(
        train_start_season=args.train_start_season,
        train_end_season=args.train_end_season,
        test_season=args.test_season,
        min_ev=args.min_ev,
        min_probability=args.min_probability,
        max_daily_picks=args.max_daily_picks,
    )
    model_path = args.model_path or args.out_dir / "ou_calibrator.joblib"
    predictions = pd.read_csv(args.predictions)
    result = run_calibration(predictions, cfg, model_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "calibration_version": CALIBRATION_VERSION,
            "generated_at_utc": utc_now_iso(),
            "predictions_hash": canonical_hash(predictions),
            "config": asdict(cfg),
            "model_path": model_path,
            "outputs": {name: canonical_hash(frame) for name, frame in result.items()},
        },
    )
    print(result["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
