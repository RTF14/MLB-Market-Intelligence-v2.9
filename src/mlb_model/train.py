from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from .config import MLBModelConfig
from .governance import canonical_hash, utc_now_iso, validate_features, write_manifest


TARGETS = ["home_score", "away_score", "total_runs", "home_run_diff"]


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = set(TARGETS + ["game_pk", "game_date", "game_type", "home_team", "away_team"])
    categorical_candidates = [
        "home_team_id",
        "away_team_id",
        "venue_id",
    ]
    categorical = [column for column in categorical_candidates if column in frame.columns]
    numeric = [
        column
        for column in frame.columns
        if column not in excluded
        and column not in categorical
        and pd.api.types.is_numeric_dtype(frame[column])
        and not frame[column].isna().all()
    ]
    return numeric, categorical


def build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=3)),
        ]
    )


def _rmse(actual: pd.Series, predictions: pd.Series) -> float:
    return math.sqrt(mean_squared_error(actual, predictions))


def train_models(
    features: pd.DataFrame,
    model_dir: Path,
    *,
    config: MLBModelConfig | None = None,
) -> pd.DataFrame:
    cfg = config or MLBModelConfig()
    features = validate_features(features)
    if len(features) < cfg.min_training_rows:
        raise ValueError(f"Need at least {cfg.min_training_rows} feature rows, got {len(features)}")
    features = features.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    split_index = int(len(features) * (1.0 - cfg.test_fraction))
    train = features.iloc[:split_index]
    test = features.iloc[split_index:]

    numeric, categorical = feature_columns(features)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, float | str]] = []

    for target in TARGETS:
        pipeline = build_pipeline(numeric, categorical)
        pipeline.fit(train[numeric + categorical], train[target])
        predictions = pipeline.predict(test[numeric + categorical])
        metrics.append(
            {
                "target": target,
                "mae": mean_absolute_error(test[target], predictions),
                "rmse": _rmse(test[target], predictions),
                "test_rows": len(test),
            }
        )
        joblib.dump(pipeline, model_dir / f"{target}.joblib")

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(model_dir / "metrics.csv", index=False)
    write_manifest(
        model_dir / "manifest.json",
        {
            "model_version": cfg.model_version,
            "generated_at_utc": utc_now_iso(),
            "input_hash": canonical_hash(features),
            "config_hash": canonical_hash(asdict(cfg)),
            "rows": len(features),
            "train_rows": len(train),
            "test_rows": len(test),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "targets": TARGETS,
            "metrics": metrics,
        },
    )
    return metrics_frame


def fit_models_on_frame(
    train: pd.DataFrame,
    model_dir: Path,
    *,
    config: MLBModelConfig | None = None,
) -> dict[str, object]:
    cfg = config or MLBModelConfig()
    train = validate_features(train)
    if len(train) < cfg.min_training_rows:
        raise ValueError(f"Need at least {cfg.min_training_rows} training rows, got {len(train)}")

    numeric, categorical = feature_columns(train)
    model_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, object] = {}
    for target in TARGETS:
        pipeline = build_pipeline(numeric, categorical)
        pipeline.fit(train[numeric + categorical], train[target])
        joblib.dump(pipeline, model_dir / f"{target}.joblib")
        models[target] = pipeline

    write_manifest(
        model_dir / "manifest.json",
        {
            "model_version": cfg.model_version,
            "generated_at_utc": utc_now_iso(),
            "input_hash": canonical_hash(train),
            "config_hash": canonical_hash(asdict(cfg)),
            "train_rows": len(train),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "targets": TARGETS,
            "train_mode": "explicit_frame",
        },
    )
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLB score and total models.")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    metrics = train_models(features, args.model_dir)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
