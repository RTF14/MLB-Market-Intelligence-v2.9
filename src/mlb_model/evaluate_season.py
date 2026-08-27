from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from .governance import canonical_hash, utc_now_iso, validate_features, write_manifest
from .train import TARGETS, feature_columns, fit_models_on_frame


def _rmse(actual: pd.Series, pred: pd.Series) -> float:
    return float(mean_squared_error(actual, pred) ** 0.5)


def _predict(models: dict[str, object], train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    numeric, categorical = feature_columns(train)
    out_cols = [
        col
        for col in [
            "season",
            "game_date",
            "game_pk",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "total_runs",
            "home_run_diff",
            "total_line",
        ]
        if col in test.columns
    ]
    out = test[out_cols].copy()
    for target in TARGETS:
        name = "pred_total" if target == "total_runs" else f"pred_{target}"
        out[name] = models[target].predict(test[numeric + categorical]).round(3)
    out["pred_margin"] = (out["pred_home_score"] - out["pred_away_score"]).round(3)
    out["pred_home_win"] = out["pred_margin"] > 0
    out["actual_home_win"] = out["home_score"] > out["away_score"]
    out["pred_winner"] = out["home_team"].where(out["pred_home_win"], out["away_team"])
    out["actual_winner"] = out["home_team"].where(out["actual_home_win"], out["away_team"])
    out["score_abs_error_home"] = (out["pred_home_score"] - out["home_score"]).abs()
    out["score_abs_error_away"] = (out["pred_away_score"] - out["away_score"]).abs()
    out["total_abs_error"] = (out["pred_total"] - out["total_runs"]).abs()
    out["margin_abs_error"] = (out["pred_margin"] - out["home_run_diff"]).abs()
    dates = pd.to_datetime(out["game_date"], errors="raise")
    iso = dates.dt.isocalendar()
    out["test_week"] = iso["week"].astype(int)
    out["week_start"] = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.date.astype(str)
    return out


def summarize_weekly(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (week, week_start), group in predictions.groupby(["test_week", "week_start"], sort=True):
        row = {
            "test_week": int(week),
            "week_start": week_start,
            "games": int(len(group)),
            "home_score_mae": float(group["score_abs_error_home"].mean()),
            "away_score_mae": float(group["score_abs_error_away"].mean()),
            "total_mae": float(group["total_abs_error"].mean()),
            "total_rmse": _rmse(group["total_runs"], group["pred_total"]),
            "margin_mae": float(group["margin_abs_error"].mean()),
            "winner_accuracy": float(accuracy_score(group["actual_home_win"], group["pred_home_win"])),
        }
        if "total_line" in group.columns and group["total_line"].notna().all():
            pred_over = group["pred_total"] > group["total_line"]
            actual_over = group["total_runs"] > group["total_line"]
            non_push = group["total_runs"] != group["total_line"]
            row["ou_accuracy_vs_line"] = (
                float(accuracy_score(actual_over[non_push], pred_over[non_push])) if non_push.any() else 0.0
            )
            row["ou_pushes"] = int((~non_push).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_overall(predictions: pd.DataFrame) -> pd.DataFrame:
    row = {
        "games": int(len(predictions)),
        "home_score_mae": mean_absolute_error(predictions["home_score"], predictions["pred_home_score"]),
        "home_score_rmse": _rmse(predictions["home_score"], predictions["pred_home_score"]),
        "away_score_mae": mean_absolute_error(predictions["away_score"], predictions["pred_away_score"]),
        "away_score_rmse": _rmse(predictions["away_score"], predictions["pred_away_score"]),
        "total_mae": mean_absolute_error(predictions["total_runs"], predictions["pred_total"]),
        "total_rmse": _rmse(predictions["total_runs"], predictions["pred_total"]),
        "margin_mae": mean_absolute_error(predictions["home_run_diff"], predictions["pred_margin"]),
        "margin_rmse": _rmse(predictions["home_run_diff"], predictions["pred_margin"]),
        "winner_accuracy": accuracy_score(predictions["actual_home_win"], predictions["pred_home_win"]),
    }
    if "total_line" in predictions.columns and predictions["total_line"].notna().all():
        pred_over = predictions["pred_total"] > predictions["total_line"]
        actual_over = predictions["total_runs"] > predictions["total_line"]
        non_push = predictions["total_runs"] != predictions["total_line"]
        row["ou_accuracy_vs_line"] = float(accuracy_score(actual_over[non_push], pred_over[non_push])) if non_push.any() else 0.0
        row["ou_pushes"] = int((~non_push).sum())
    return pd.DataFrame([row])


def evaluate_season(
    features: pd.DataFrame,
    train_end_season: int,
    test_season: int,
    model_dir: Path,
    *,
    game_type: str | None = "R",
) -> dict[str, pd.DataFrame]:
    features = validate_features(features)
    if game_type and "game_type" in features.columns:
        features = features[features["game_type"].astype(str).eq(game_type)].copy()
    train = features[features["season"] <= train_end_season].copy()
    test = features[features["season"] == test_season].copy()
    if test.empty:
        raise ValueError(f"No rows found for test season {test_season}")
    models = fit_models_on_frame(train, model_dir)
    predictions = _predict(models, train, test)
    weekly = summarize_weekly(predictions)
    overall = summarize_overall(predictions)
    return {"overall": overall, "weekly": weekly, "predictions": predictions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train through one MLB season and evaluate another by week.")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--train-end-season", default=2024, type=int)
    parser.add_argument("--test-season", default=2025, type=int)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--game-type", default="R", help="MLB game type to evaluate; use ALL to disable filtering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    game_type = None if str(args.game_type).upper() == "ALL" else args.game_type
    result = evaluate_season(features, args.train_end_season, args.test_season, args.model_dir, game_type=game_type)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "generated_at_utc": utc_now_iso(),
            "features_hash": canonical_hash(features),
            "train_end_season": args.train_end_season,
            "test_season": args.test_season,
            "game_type": game_type,
            "outputs": {name: canonical_hash(frame) for name, frame in result.items()},
        },
    )
    print(result["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
