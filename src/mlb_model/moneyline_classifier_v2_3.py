from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .governance import canonical_hash, utc_now_iso, write_manifest
from .moneyline_classifier_v2_2 import american_implied_probability, american_profit_per_unit


RESEARCH_VERSION = "mlb_moneyline_classifier_v2_3"


@dataclass(frozen=True)
class MoneylineV23Config:
    train_start_season: int = 2021
    train_end_season: int = 2025
    test_season: int = 2026
    min_model_probability: float = 0.52
    research_version: str = RESEARCH_VERSION


SIDE_NUMERIC_FEATURES = [
    "side_is_home",
    "is_favorite",
    "selected_price",
    "selected_implied_probability",
    "selected_no_vig_probability",
    "market_vig",
    "model_margin_for_side",
    "model_total",
    "rest_days_diff",
    "games_last_3_days_diff",
    "travel_flag_diff",
    "starter_rest_days_diff",
    "starter_recent_form_diff_for_side",
    "starter_xFIP_diff_for_side",
    "starter_kbb_diff_for_side",
    "offense_wRC_plus_diff_for_side",
    "team_offense_form_diff_l7_for_side",
    "team_offense_form_diff_l14_for_side",
    "team_offense_form_diff_l30_for_side",
    "bullpen_xFIP_diff_for_side",
    "bullpen_fatigue_diff_for_side",
    "park_run_factor",
]

SIDE_CATEGORICAL_FEATURES = ["side", "favorite_group", "price_band", "venue_id"]


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _price_band(price: pd.Series) -> pd.Series:
    p = pd.to_numeric(price, errors="coerce")
    out = pd.Series("UNKNOWN", index=price.index, dtype=object)
    out.loc[p.between(-159.999, -100.0, inclusive="both")] = "FAV_LT_160"
    out.loc[p.between(-219.999, -160.0, inclusive="both")] = "FAV_160_220"
    out.loc[p <= -220.0] = "FAV_220_PLUS"
    out.loc[p.between(100.0, 139.999, inclusive="both")] = "DOG_100_140"
    out.loc[p >= 140.0] = "DOG_140_PLUS"
    return out


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def merge_feature_context(predictions: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
    if features is None:
        return predictions.copy()
    needed = {
        "game_pk",
        "home_rest_days",
        "away_rest_days",
        "home_games_last_3_days",
        "away_games_last_3_days",
        "home_travel_flag",
        "away_travel_flag",
        "home_starter_rest_days",
        "away_starter_rest_days",
        "home_starter_recent_form_index",
        "away_starter_recent_form_index",
        "home_sp_xFIP",
        "away_sp_xFIP",
        "home_sp_kbb",
        "away_sp_kbb",
        "home_wRC_plus_vs_hand",
        "away_wRC_plus_vs_hand",
        "bullpen_xFIP_diff",
        "bullpen_fatigue_diff",
        "starter_xFIP_diff",
        "starter_kbb_diff",
        "offense_wRC_plus_diff",
        "team_offense_form_diff_l7",
        "team_offense_form_diff_l14",
        "team_offense_form_diff_l30",
        "starter_recent_form_diff",
        "park_run_factor",
        "venue_id",
    }
    extras = features[[col for col in features.columns if col in needed]].drop_duplicates("game_pk")
    out = predictions.merge(extras, on="game_pk", how="left", suffixes=("", "_feature"))
    for col in needed - {"game_pk"}:
        feature_col = f"{col}_feature"
        if feature_col in out.columns:
            if col not in out.columns:
                out[col] = out[feature_col]
            else:
                out[col] = out[col].combine_first(out[feature_col])
            out = out.drop(columns=[feature_col])
    return out


def build_candidate_frame(predictions: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
    required = {"season", "game_date", "game_pk", "home_score", "away_score", "home_moneyline", "away_moneyline"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Moneyline v2.3 input missing columns: {sorted(missing)}")
    games = merge_feature_context(predictions, features)
    games["season"] = pd.to_numeric(games["season"], errors="raise").astype(int)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="raise").dt.date.astype(str)
    games["actual_home_win"] = _num(games, "home_score") > _num(games, "away_score")
    games["home_moneyline"] = _num(games, "home_moneyline")
    games["away_moneyline"] = _num(games, "away_moneyline")
    games["home_moneyline_open"] = _num(games, "home_moneyline_open")
    games["away_moneyline_open"] = _num(games, "away_moneyline_open")
    games = games.dropna(subset=["home_moneyline", "away_moneyline"]).copy()
    home_imp = american_implied_probability(games["home_moneyline"])
    away_imp = american_implied_probability(games["away_moneyline"])
    vig_sum = home_imp + away_imp
    games["market_vig"] = vig_sum - 1.0
    games["home_no_vig_probability"] = _safe_divide(home_imp, vig_sum)
    games["away_no_vig_probability"] = _safe_divide(away_imp, vig_sum)

    def make_side(side: str) -> pd.DataFrame:
        selected_home = side == "HOME"
        out = games.copy()
        out["side"] = side
        out["side_is_home"] = 1 if selected_home else 0
        out["display_side"] = out["home_team"] if selected_home and "home_team" in out else out.get("away_team", side)
        if not selected_home and "away_team" in out:
            out["display_side"] = out["away_team"]
        out["selected_price"] = out["home_moneyline"] if selected_home else out["away_moneyline"]
        out["opponent_price"] = out["away_moneyline"] if selected_home else out["home_moneyline"]
        out["selected_open_price"] = out["home_moneyline_open"] if selected_home else out["away_moneyline_open"]
        out["selected_implied_probability"] = american_implied_probability(out["selected_price"])
        out["selected_open_implied_probability"] = american_implied_probability(out["selected_open_price"])
        out["selected_no_vig_probability"] = out["home_no_vig_probability"] if selected_home else out["away_no_vig_probability"]
        out["actual_result"] = np.where(out["actual_home_win"].eq(selected_home), "WIN", "LOSS")
        out["target_side_win"] = out["actual_result"].eq("WIN").astype(int)
        sign = 1.0 if selected_home else -1.0
        out["model_margin_for_side"] = sign * _num(out, "pred_margin")
        out["model_total"] = _num(out, "pred_total")
        out["rest_days_diff"] = sign * (_num(out, "home_rest_days") - _num(out, "away_rest_days"))
        out["games_last_3_days_diff"] = sign * (_num(out, "home_games_last_3_days") - _num(out, "away_games_last_3_days"))
        out["travel_flag_diff"] = sign * (_num(out, "home_travel_flag") - _num(out, "away_travel_flag"))
        out["starter_rest_days_diff"] = sign * (_num(out, "home_starter_rest_days") - _num(out, "away_starter_rest_days"))
        out["starter_recent_form_diff_for_side"] = sign * _num(out, "starter_recent_form_diff")
        out["starter_xFIP_diff_for_side"] = -sign * _num(out, "starter_xFIP_diff")
        out["starter_kbb_diff_for_side"] = sign * _num(out, "starter_kbb_diff")
        out["offense_wRC_plus_diff_for_side"] = sign * _num(out, "offense_wRC_plus_diff")
        out["team_offense_form_diff_l7_for_side"] = sign * _num(out, "team_offense_form_diff_l7")
        out["team_offense_form_diff_l14_for_side"] = sign * _num(out, "team_offense_form_diff_l14")
        out["team_offense_form_diff_l30_for_side"] = sign * _num(out, "team_offense_form_diff_l30")
        out["bullpen_xFIP_diff_for_side"] = -sign * _num(out, "bullpen_xFIP_diff")
        out["bullpen_fatigue_diff_for_side"] = -sign * _num(out, "bullpen_fatigue_diff")
        out["park_run_factor"] = _num(out, "park_run_factor", 1.0)
        out["payout_per_unit"] = american_profit_per_unit(out["selected_price"])
        out["price_band"] = _price_band(out["selected_price"])
        out["is_favorite"] = out["selected_no_vig_probability"].ge(0.5).astype(int)
        out["favorite_group"] = np.where(out["is_favorite"].eq(1), "FAVORITE", "UNDERDOG")
        out["clv_result"] = "MISSING_OPEN"
        has_open = out["selected_open_implied_probability"].notna()
        out.loc[has_open, "clv_result"] = "PUSH"
        out.loc[has_open & (out["selected_implied_probability"] > out["selected_open_implied_probability"] + 0.0025), "clv_result"] = "WIN"
        out.loc[has_open & (out["selected_implied_probability"] < out["selected_open_implied_probability"] - 0.0025), "clv_result"] = "LOSS"
        return out

    candidates = pd.concat([make_side("HOME"), make_side("AWAY")], ignore_index=True, sort=False)
    candidates = candidates.dropna(subset=["selected_price", "payout_per_unit", "selected_no_vig_probability"]).copy()
    for col in SIDE_CATEGORICAL_FEATURES:
        if col not in candidates:
            candidates[col] = ""
        candidates[col] = candidates[col].astype(str)
    for col in SIDE_NUMERIC_FEATURES:
        if col not in candidates:
            candidates[col] = np.nan
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    return candidates.sort_values(["game_date", "game_pk", "side"], kind="mergesort").reset_index(drop=True)


def build_pipeline() -> CalibratedClassifierCV:
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())])
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, SIDE_NUMERIC_FEATURES),
            ("categorical", categorical, SIDE_CATEGORICAL_FEATURES),
        ]
    )
    base = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def fit_models(frame: pd.DataFrame, cfg: MoneylineV23Config, model_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    train = frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)].copy()
    if train.empty:
        raise ValueError("No v2.3 moneyline training rows")
    model_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, object] = {}
    metadata = []
    for group_name, group in train.groupby("favorite_group", sort=True):
        if group["target_side_win"].nunique() < 2:
            continue
        model = build_pipeline()
        model.fit(group[SIDE_NUMERIC_FEATURES + SIDE_CATEGORICAL_FEATURES], group["target_side_win"])
        path = model_dir / f"{group_name.lower()}_moneyline_v2_3.joblib"
        joblib.dump(model, path)
        models[group_name] = model
        metadata.append(
            {
                "favorite_group": group_name,
                "model_path": str(path),
                "training_rows": int(len(group)),
                "training_hash": canonical_hash(group),
                "target": "target_side_win",
                "sklearn_version": sklearn.__version__,
            }
        )
    if not models:
        raise ValueError("No v2.3 favorite/underdog models were trained")
    return models, pd.DataFrame(metadata)


def score_candidates(frame: pd.DataFrame, models: dict[str, object], cfg: MoneylineV23Config) -> pd.DataFrame:
    out = frame.copy()
    out["model_probability"] = np.nan
    for group_name, model in models.items():
        mask = out["favorite_group"].eq(group_name)
        if mask.any():
            out.loc[mask, "model_probability"] = model.predict_proba(out.loc[mask, SIDE_NUMERIC_FEATURES + SIDE_CATEGORICAL_FEATURES])[:, 1]
    out["probability_edge"] = out["model_probability"] - out["selected_no_vig_probability"]
    out["ev"] = out["model_probability"] * out["payout_per_unit"] - (1.0 - out["model_probability"])
    out["rank_score"] = (out["ev"] * 100.0 + out["probability_edge"] + out["model_probability"] / 10.0).round(6)
    out["profit_units"] = 0.0
    out.loc[out["actual_result"].eq("WIN"), "profit_units"] = out.loc[out["actual_result"].eq("WIN"), "payout_per_unit"]
    out.loc[out["actual_result"].eq("LOSS"), "profit_units"] = -1.0
    return out


def _pick_orders(scored: pd.DataFrame, *, fav_edge: float, dog_edge: float, min_ev: float, max_daily: int, test_season: int, min_probability: float) -> pd.DataFrame:
    test = scored[scored["season"].eq(test_season)].copy()
    edge_threshold = np.where(test["favorite_group"].eq("FAVORITE"), fav_edge, dog_edge)
    eligible = (
        test["model_probability"].ge(min_probability)
        & test["probability_edge"].ge(edge_threshold)
        & test["ev"].ge(min_ev)
    )
    pool = test[eligible].copy()
    game_best = []
    for _game_pk, group in pool.groupby("game_pk", sort=False):
        game_best.append(group.sort_values(["rank_score", "ev", "probability_edge", "game_pk", "side"], ascending=[False, False, False, True, True]).head(1))
    pool = pd.concat(game_best, ignore_index=False, sort=False) if game_best else pd.DataFrame()
    parts = []
    for _date, group in pool.groupby("game_date", sort=True):
        parts.append(group.sort_values(["rank_score", "ev", "probability_edge", "game_pk", "side"], ascending=[False, False, False, True, True]).head(max_daily))
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame(columns=scored.columns)


def _summary_row(df: pd.DataFrame, label: str, params: dict | None = None) -> dict:
    wins = int(df["actual_result"].eq("WIN").sum()) if not df.empty else 0
    losses = int(df["actual_result"].eq("LOSS").sum()) if not df.empty else 0
    pushes = int(df["actual_result"].eq("PUSH").sum()) if "actual_result" in df else 0
    decisions = wins + losses
    profit = float(df["profit_units"].sum()) if "profit_units" in df and not df.empty else 0.0
    clv_decisions = df["clv_result"].isin(["WIN", "LOSS"]) if "clv_result" in df and not df.empty else pd.Series(dtype=bool)
    row = {
        "test": label,
        "bets": int(len(df)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": wins / decisions if decisions else 0.0,
        "profit_units": profit,
        "roi": profit / len(df) if len(df) else 0.0,
        "clv_wins": int(df["clv_result"].eq("WIN").sum()) if "clv_result" in df and not df.empty else 0,
        "clv_losses": int(df["clv_result"].eq("LOSS").sum()) if "clv_result" in df and not df.empty else 0,
        "clv_win_rate": float(df.loc[clv_decisions, "clv_result"].eq("WIN").mean()) if len(clv_decisions) and clv_decisions.any() else 0.0,
        "avg_probability_edge": float(df["probability_edge"].mean()) if "probability_edge" in df and not df.empty else 0.0,
        "avg_ev": float(df["ev"].mean()) if "ev" in df and not df.empty else 0.0,
    }
    if params:
        row.update(params)
    return row


def grid_search(scored: pd.DataFrame, cfg: MoneylineV23Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    order_map = {}
    for fav_edge in [0.03, 0.04, 0.05, 0.07]:
        for dog_edge in [0.03, 0.05, 0.06, 0.08]:
            for min_ev in [0.0, 0.02, 0.04, 0.06]:
                for max_daily in [1, 2, 3, 4]:
                    orders = _pick_orders(
                        scored,
                        fav_edge=fav_edge,
                        dog_edge=dog_edge,
                        min_ev=min_ev,
                        max_daily=max_daily,
                        test_season=cfg.test_season,
                        min_probability=cfg.min_model_probability,
                    )
                    key = f"fav{fav_edge:g}_dog{dog_edge:g}_ev{min_ev:g}_k{max_daily}"
                    rows.append(
                        _summary_row(
                            orders,
                            key,
                            {"fav_edge": fav_edge, "dog_edge": dog_edge, "min_ev": min_ev, "max_daily": max_daily},
                        )
                    )
                    order_map[key] = orders
    grid = pd.DataFrame(rows).sort_values(["roi", "profit_units", "bets"], ascending=[False, False, False]).reset_index(drop=True)
    best_key = grid.iloc[0]["test"] if not grid.empty else ""
    return grid, order_map.get(best_key, pd.DataFrame(columns=scored.columns))


def price_band_report(scored: pd.DataFrame, cfg: MoneylineV23Config) -> pd.DataFrame:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    rows = []
    for (band, group_name), group in test.groupby(["price_band", "favorite_group"], sort=True):
        rows.append(_summary_row(group, f"{group_name}_{band}", {"price_band": band, "favorite_group": group_name}))
    return pd.DataFrame(rows).sort_values(["favorite_group", "price_band"]).reset_index(drop=True)


def model_metrics(scored: pd.DataFrame, cfg: MoneylineV23Config) -> pd.DataFrame:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    rows = []
    for group_name, group in test.groupby("favorite_group", sort=True):
        y = group["target_side_win"].astype(int)
        p = group["model_probability"]
        row = {
            "favorite_group": group_name,
            "rows": int(len(group)),
            "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p, labels=[0, 1]),
            "accuracy_at_50": accuracy_score(y, p >= 0.5),
        }
        if y.nunique() > 1:
            row["auc"] = roc_auc_score(y, p)
        rows.append(row)
    return pd.DataFrame(rows)


def run(predictions: pd.DataFrame, features: pd.DataFrame | None, cfg: MoneylineV23Config, model_dir: Path) -> dict[str, pd.DataFrame]:
    frame = build_candidate_frame(predictions, features)
    models, metadata = fit_models(frame, cfg, model_dir)
    scored = score_candidates(frame, models, cfg)
    grid, best_orders = grid_search(scored, cfg)
    best_summary = pd.DataFrame([_summary_row(best_orders, "best_grid")])
    return {
        "scored_candidates": scored,
        "grid_search": grid,
        "best_orders": best_orders.reset_index(drop=True),
        "best_summary": best_summary,
        "price_band_report": price_band_report(scored, cfg),
        "model_metrics": model_metrics(scored, cfg),
        "model_metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLB moneyline v2.3: side-level favorite/underdog classifiers and gate search.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--train-start-season", default=MoneylineV23Config.train_start_season, type=int)
    parser.add_argument("--train-end-season", default=MoneylineV23Config.train_end_season, type=int)
    parser.add_argument("--test-season", default=MoneylineV23Config.test_season, type=int)
    parser.add_argument("--min-model-probability", default=MoneylineV23Config.min_model_probability, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MoneylineV23Config(
        train_start_season=args.train_start_season,
        train_end_season=args.train_end_season,
        test_season=args.test_season,
        min_model_probability=args.min_model_probability,
    )
    predictions = pd.read_csv(args.predictions)
    features = pd.read_csv(args.features) if args.features else None
    model_dir = args.model_dir or args.out_dir / "models"
    outputs = run(predictions, features, cfg, model_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "research_version": RESEARCH_VERSION,
            "generated_at_utc": utc_now_iso(),
            "predictions_hash": canonical_hash(predictions),
            "features_hash": canonical_hash(features) if features is not None else None,
            "config": asdict(cfg),
            "outputs": {name: canonical_hash(frame) for name, frame in outputs.items()},
        },
    )
    print(outputs["best_summary"].to_string(index=False))
    print(outputs["grid_search"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
