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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from .governance import canonical_hash, utc_now_iso, write_manifest
from .synthetic_totals import add_synthetic_total_line


CALIBRATION_VERSION = "mlb_ou_calibration_v1_7"


@dataclass(frozen=True)
class OUCalibrationV17Config:
    train_start_season: int = 2021
    train_end_season: int = 2024
    test_season: int = 2025
    min_abs_edge: float = 0.75
    max_abs_edge: float = 8.0
    min_ev: float = 0.0
    min_probability: float = 0.525
    max_daily_picks: int = 4
    max_vig: float = 0.08
    calibration_version: str = CALIBRATION_VERSION


FEATURE_COLUMNS_NUMERIC_V17 = [
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
    "total_line_open",
    "total_line_move",
    "selected_price",
    "selected_price_open",
    "selected_implied_probability",
    "market_vig",
    "market_vig_open",
]

FEATURE_COLUMNS_CATEGORICAL_V17 = ["side"]


def american_profit_per_unit(price: pd.Series) -> pd.Series:
    odds = pd.to_numeric(price, errors="coerce")
    profit = pd.Series(np.nan, index=price.index, dtype=float)
    valid = odds.abs() >= 80
    pos = valid & (odds > 0)
    neg = valid & (odds < 0)
    profit.loc[pos] = odds.loc[pos] / 100.0
    profit.loc[neg] = 100.0 / odds.loc[neg].abs()
    return profit


def american_implied_probability(price: pd.Series) -> pd.Series:
    odds = pd.to_numeric(price, errors="coerce")
    implied = pd.Series(np.nan, index=price.index, dtype=float)
    pos = odds >= 80
    neg = odds <= -80
    implied.loc[pos] = 100.0 / (odds.loc[pos] + 100.0)
    implied.loc[neg] = odds.loc[neg].abs() / (odds.loc[neg].abs() + 100.0)
    return implied


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def ensure_synthetic_preserving_market(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    if "market_total_line" not in out.columns and "total_line" in out.columns:
        out["market_total_line"] = pd.to_numeric(out["total_line"], errors="coerce")
    if "synthetic_total_line" not in out.columns:
        out = add_synthetic_total_line(out)
    if "market_total_line" in out.columns:
        out["total_line"] = out["market_total_line"]
    return out


def build_ou_training_frame_v1_7(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "game_date", "game_pk", "pred_total", "total_line", "total_runs"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"O/U v1.7 calibration input missing columns: {sorted(missing)}")

    out = ensure_synthetic_preserving_market(predictions)
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise")
    out["month"] = out["game_date"].dt.month.astype(int)
    out["model_total"] = _num(out, "pred_total")
    out["market_total"] = _num(out, "total_line")
    out["actual_total"] = _num(out, "total_runs")
    out["synthetic_total"] = _num(out, "synthetic_total_line")
    out["total_line_open"] = _num(out, "total_line_open")
    out["total_line_move"] = _num(out, "total_line_move")
    if out["total_line_move"].isna().all() and "total_line_open" in out.columns:
        out["total_line_move"] = out["market_total"] - out["total_line_open"]
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

    over_imp = american_implied_probability(_num(out, "total_price_over"))
    under_imp = american_implied_probability(_num(out, "total_price_under"))
    over_imp_open = american_implied_probability(_num(out, "total_price_over_open"))
    under_imp_open = american_implied_probability(_num(out, "total_price_under_open"))
    out["market_vig"] = over_imp + under_imp - 1.0
    out["market_vig_open"] = over_imp_open + under_imp_open - 1.0

    over = out.copy()
    over["side"] = "OVER"
    over["edge"] = over["model_minus_market"]
    over["target_win"] = over["actual_over"]
    over["selected_price"] = _num(over, "total_price_over")
    over["selected_price_open"] = _num(over, "total_price_over_open")

    under = out.copy()
    under["side"] = "UNDER"
    under["edge"] = -under["model_minus_market"]
    under["target_win"] = under["actual_under"]
    under["selected_price"] = _num(under, "total_price_under")
    under["selected_price_open"] = _num(under, "total_price_under_open")

    stacked = pd.concat([over, under], ignore_index=True, sort=False)
    stacked = stacked[~stacked["push"]].copy()
    stacked = stacked[stacked["edge"] > 0].copy()
    stacked["abs_edge"] = stacked["edge"].abs()
    stacked["selected_implied_probability"] = american_implied_probability(stacked["selected_price"])
    stacked["payout_per_unit"] = american_profit_per_unit(stacked["selected_price"])
    stacked = stacked.dropna(subset=["selected_price", "selected_implied_probability", "payout_per_unit"])
    return stacked.sort_values(["game_date", "game_pk", "side"], kind="mergesort").reset_index(drop=True)


def build_pipeline() -> Pipeline:
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, FEATURE_COLUMNS_NUMERIC_V17),
            ("categorical", categorical, FEATURE_COLUMNS_CATEGORICAL_V17),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def fit_calibrator(frame: pd.DataFrame, cfg: OUCalibrationV17Config, model_path: Path) -> tuple[Pipeline, pd.DataFrame]:
    train = frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)].copy()
    if train.empty:
        raise ValueError("No v1.7 calibration training rows after filtering")
    model = build_pipeline()
    model.fit(train[FEATURE_COLUMNS_NUMERIC_V17 + FEATURE_COLUMNS_CATEGORICAL_V17], train["target_win"].astype(int))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {
        "calibration_version": cfg.calibration_version,
        "model_type": "sklearn.pipeline.Pipeline(LogisticRegression)",
        "sklearn_version": sklearn.__version__,
        "training_data_hash": canonical_hash(train),
        "feature_schema": {
            "numeric": FEATURE_COLUMNS_NUMERIC_V17,
            "categorical": FEATURE_COLUMNS_CATEGORICAL_V17,
            "all": FEATURE_COLUMNS_NUMERIC_V17 + FEATURE_COLUMNS_CATEGORICAL_V17,
        },
        "config": asdict(cfg),
        "generated_at_utc": utc_now_iso(),
    }
    write_manifest(model_path.with_suffix(".metadata.json"), metadata)
    return model, train


def score_calibrator(model: Pipeline, frame: pd.DataFrame, cfg: OUCalibrationV17Config) -> pd.DataFrame:
    out = frame.copy()
    out["calibrated_probability"] = model.predict_proba(
        out[FEATURE_COLUMNS_NUMERIC_V17 + FEATURE_COLUMNS_CATEGORICAL_V17]
    )[:, 1]
    out["calibrated_ev"] = out["calibrated_probability"] * out["payout_per_unit"] - (1.0 - out["calibrated_probability"])
    out["eligible"] = (
        (out["abs_edge"] >= cfg.min_abs_edge)
        & (out["abs_edge"] <= cfg.max_abs_edge)
        & (out["calibrated_probability"] >= cfg.min_probability)
        & (out["calibrated_ev"] >= cfg.min_ev)
        & (out["market_vig"].isna() | (out["market_vig"] <= cfg.max_vig))
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


def summarize(scored: pd.DataFrame, cfg: OUCalibrationV17Config) -> dict[str, pd.DataFrame]:
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
    return {"overall": overall, "scored_candidates": scored}


def model_metrics(scored: pd.DataFrame, cfg: OUCalibrationV17Config) -> pd.DataFrame:
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


def run_calibration(predictions: pd.DataFrame, cfg: OUCalibrationV17Config, model_path: Path) -> dict[str, pd.DataFrame]:
    frame = build_ou_training_frame_v1_7(predictions)
    model, train = fit_calibrator(frame, cfg, model_path)
    scored = score_calibrator(model, frame, cfg)
    result = summarize(scored, cfg)
    result["model_metrics"] = model_metrics(scored, cfg)
    result["training_rows"] = pd.DataFrame([{"rows": len(train), "hash": canonical_hash(train)}])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLB O/U v1.7 calibration on preserved market totals and line movement.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--train-start-season", default=OUCalibrationV17Config.train_start_season, type=int)
    parser.add_argument("--train-end-season", default=OUCalibrationV17Config.train_end_season, type=int)
    parser.add_argument("--test-season", default=OUCalibrationV17Config.test_season, type=int)
    parser.add_argument("--min-ev", default=OUCalibrationV17Config.min_ev, type=float)
    parser.add_argument("--min-probability", default=OUCalibrationV17Config.min_probability, type=float)
    parser.add_argument("--max-daily-picks", default=OUCalibrationV17Config.max_daily_picks, type=int)
    parser.add_argument("--max-vig", default=OUCalibrationV17Config.max_vig, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OUCalibrationV17Config(
        train_start_season=args.train_start_season,
        train_end_season=args.train_end_season,
        test_season=args.test_season,
        min_ev=args.min_ev,
        min_probability=args.min_probability,
        max_daily_picks=args.max_daily_picks,
        max_vig=args.max_vig,
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
            "model_path": str(model_path),
            "outputs": {name: canonical_hash(frame) for name, frame in result.items()},
        },
    )
    print(result["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
