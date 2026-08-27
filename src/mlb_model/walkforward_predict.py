from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .governance import canonical_hash, utc_now_iso, validate_features, write_manifest
from .train import TARGETS, feature_columns, fit_models_on_frame


def walkforward_predictions(
    features: pd.DataFrame,
    *,
    start_season: int,
    end_season: int,
    model_root: Path,
    game_type: str | None = "R",
) -> pd.DataFrame:
    features = validate_features(features)
    if game_type and "game_type" in features.columns:
        features = features[features["game_type"].astype(str).eq(game_type)].copy()
    features = features.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    parts = []
    for season in range(start_season, end_season + 1):
        train = features[features["season"] < season].copy()
        test = features[features["season"] == season].copy()
        if train.empty or test.empty:
            continue
        season_model_dir = model_root / f"train_through_{season - 1}"
        models = fit_models_on_frame(train, season_model_dir)
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
                "home_team_id",
                "away_team_id",
                "venue_id",
                "home_score",
                "away_score",
                "total_runs",
                "home_run_diff",
            ]
            if col in test.columns
        ]
        out = test[out_cols].copy()
        for target in TARGETS:
            name = "pred_total" if target == "total_runs" else f"pred_{target}"
            out[name] = models[target].predict(test[numeric + categorical]).round(3)
        out["pred_margin"] = (out["pred_home_score"] - out["pred_away_score"]).round(3)
        out["model_train_end_season"] = season - 1
        parts.append(out)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False).sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build walk-forward MLB prediction panel.")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--start-season", required=True, type=int)
    parser.add_argument("--end-season", required=True, type=int)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--game-type", default="R")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game_type = None if str(args.game_type).upper() == "ALL" else args.game_type
    features = pd.read_csv(args.features)
    preds = walkforward_predictions(
        features,
        start_season=args.start_season,
        end_season=args.end_season,
        model_root=args.model_root,
        game_type=game_type,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out, index=False)
    write_manifest(
        args.out.with_suffix(".manifest.json"),
        {
            "generated_at_utc": utc_now_iso(),
            "features_hash": canonical_hash(features),
            "output_hash": canonical_hash(preds),
            "rows": len(preds),
            "start_season": args.start_season,
            "end_season": args.end_season,
            "game_type": game_type,
        },
    )
    print(f"Wrote {len(preds)} walk-forward predictions to {args.out}")


if __name__ == "__main__":
    main()
