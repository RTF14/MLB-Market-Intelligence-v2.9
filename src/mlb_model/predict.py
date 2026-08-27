from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .governance import canonical_hash, utc_now_iso
from .train import TARGETS, feature_columns


def load_models(model_dir: Path) -> dict[str, object]:
    return {target: joblib.load(model_dir / f"{target}.joblib") for target in TARGETS}


def predict_scores(features: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    models = load_models(model_dir)
    numeric, categorical = feature_columns(features)
    out_cols = [
        col
        for col in [
            "season",
            "game_date",
            "game_pk",
            "home_team",
            "away_team",
            "home_team_id",
            "away_team_id",
            "venue_id",
            "total_line",
            "home_score",
            "away_score",
            "total_runs",
            "home_run_diff",
        ]
        if col in features.columns
    ]
    out = features[out_cols].copy()
    for target, model in models.items():
        name = "pred_total" if target == "total_runs" else f"pred_{target}"
        out[name] = model.predict(features[numeric + categorical]).round(2)
    if {"pred_home_score", "pred_away_score"}.issubset(out.columns):
        out["pred_margin"] = (out["pred_home_score"] - out["pred_away_score"]).round(2)
    out["prediction_hash"] = canonical_hash(out)
    out["generated_at_utc"] = utc_now_iso()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MLB score and total predictions.")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    preds = predict_scores(features, args.model_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out, index=False)
    print(f"Wrote {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()
