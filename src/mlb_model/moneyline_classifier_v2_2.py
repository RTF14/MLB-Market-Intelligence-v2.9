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


RESEARCH_VERSION = "mlb_moneyline_classifier_v2_2"


@dataclass(frozen=True)
class MoneylineClassifierConfig:
    train_start_season: int = 2021
    train_end_season: int = 2025
    test_season: int = 2026
    min_model_probability: float = 0.535
    min_probability_edge: float = 0.015
    min_ev: float = 0.0
    max_daily_picks: int = 4
    require_positive_ev: bool = True
    research_version: str = RESEARCH_VERSION


NUMERIC_FEATURES = [
    "home_moneyline_implied_probability",
    "away_moneyline_implied_probability",
    "home_moneyline_no_vig_probability",
    "away_moneyline_no_vig_probability",
    "moneyline_vig",
    "home_rest_days",
    "away_rest_days",
    "home_games_last_3_days",
    "away_games_last_3_days",
    "home_travel_flag",
    "away_travel_flag",
    "home_pitcher_starts",
    "away_pitcher_starts",
    "home_starter_rest_days",
    "away_starter_rest_days",
    "home_starter_team_ra_l3",
    "away_starter_team_ra_l3",
    "home_starter_team_run_diff_l3",
    "away_starter_team_run_diff_l3",
    "home_starter_team_win_rate_l3",
    "away_starter_team_win_rate_l3",
    "home_starter_recent_form_index",
    "away_starter_recent_form_index",
    "home_pitcher_team_ra_l5",
    "away_pitcher_team_ra_l5",
    "home_pitcher_team_run_diff_l5",
    "away_pitcher_team_run_diff_l5",
    "home_pitcher_team_ra_l10",
    "away_pitcher_team_ra_l10",
    "home_pitcher_team_run_diff_l10",
    "away_pitcher_team_run_diff_l10",
    "home_pitcher_team_ra_l20",
    "away_pitcher_team_ra_l20",
    "home_pitcher_team_run_diff_l20",
    "away_pitcher_team_run_diff_l20",
    "home_runs_scored_l7",
    "away_runs_scored_l7",
    "home_runs_allowed_l7",
    "away_runs_allowed_l7",
    "home_run_diff_l7",
    "away_run_diff_l7",
    "home_offense_index_l7",
    "away_offense_index_l7",
    "home_run_prevention_index_l7",
    "away_run_prevention_index_l7",
    "home_runs_scored_l14",
    "away_runs_scored_l14",
    "home_runs_allowed_l14",
    "away_runs_allowed_l14",
    "home_run_diff_l14",
    "away_run_diff_l14",
    "home_offense_index_l14",
    "away_offense_index_l14",
    "home_run_prevention_index_l14",
    "away_run_prevention_index_l14",
    "home_runs_scored_l30",
    "away_runs_scored_l30",
    "home_runs_allowed_l30",
    "away_runs_allowed_l30",
    "home_run_diff_l30",
    "away_run_diff_l30",
    "home_offense_index_l30",
    "away_offense_index_l30",
    "home_run_prevention_index_l30",
    "away_run_prevention_index_l30",
    "home_offense_momentum_7v30",
    "away_offense_momentum_7v30",
    "home_run_prevention_momentum_7v30",
    "away_run_prevention_momentum_7v30",
    "home_bullpen_stress_proxy_l3d",
    "away_bullpen_stress_proxy_l3d",
    "home_bullpen_decay_stress_proxy_l3d",
    "away_bullpen_decay_stress_proxy_l3d",
    "home_bullpen_ip_proxy_l3d",
    "away_bullpen_ip_proxy_l3d",
    "home_bullpen_fatigue_rate_l3d",
    "away_bullpen_fatigue_rate_l3d",
    "home_sp_xFIP",
    "away_sp_xFIP",
    "home_sp_kbb",
    "away_sp_kbb",
    "home_wRC_plus_vs_hand",
    "away_wRC_plus_vs_hand",
    "bullpen_xFIP_diff",
    "bullpen_fatigue_index",
    "bullpen_fatigue_diff",
    "starter_xFIP_diff",
    "starter_kbb_diff",
    "offense_wRC_plus_diff",
    "team_offense_form_diff_l7",
    "team_offense_form_diff_l14",
    "team_offense_form_diff_l30",
    "starter_recent_form_diff",
    "park_games_tracked",
    "park_total_runs_l50",
    "park_home_score_l50",
    "park_run_factor",
]

CATEGORICAL_FEATURES = ["home_team_id", "away_team_id", "venue_id"]


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


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


def merge_feature_context(predictions: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
    if features is None:
        return predictions.copy()
    feature_cols = [
        col
        for col in features.columns
        if col == "game_pk" or col in set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    ]
    extras = features[feature_cols].drop_duplicates("game_pk")
    out = predictions.merge(extras, on="game_pk", how="left", suffixes=("", "_feature"))
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        feature_col = f"{col}_feature"
        if feature_col in out.columns:
            if col not in out.columns:
                out[col] = out[feature_col]
            else:
                out[col] = out[col].combine_first(out[feature_col])
            out = out.drop(columns=[feature_col])
    return out


def build_moneyline_frame(predictions: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
    required = {"season", "game_date", "game_pk", "home_score", "away_score", "home_moneyline", "away_moneyline"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Moneyline input missing columns: {sorted(missing)}")

    out = merge_feature_context(predictions, features)
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise").dt.date.astype(str)
    out["actual_home_win"] = _num(out, "home_score") > _num(out, "away_score")
    out["home_moneyline"] = _num(out, "home_moneyline")
    out["away_moneyline"] = _num(out, "away_moneyline")
    out["home_moneyline_implied_probability"] = american_implied_probability(out["home_moneyline"])
    out["away_moneyline_implied_probability"] = american_implied_probability(out["away_moneyline"])
    implied_sum = out["home_moneyline_implied_probability"] + out["away_moneyline_implied_probability"]
    out["moneyline_vig"] = implied_sum - 1.0
    out["home_moneyline_no_vig_probability"] = out["home_moneyline_implied_probability"] / implied_sum
    out["away_moneyline_no_vig_probability"] = out["away_moneyline_implied_probability"] / implied_sum
    out = out.dropna(
        subset=[
            "home_moneyline",
            "away_moneyline",
            "home_moneyline_implied_probability",
            "away_moneyline_implied_probability",
            "home_moneyline_no_vig_probability",
            "away_moneyline_no_vig_probability",
        ]
    ).copy()
    for col in NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str)
    return out.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)


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
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    base = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def fit_classifier(frame: pd.DataFrame, cfg: MoneylineClassifierConfig, model_dir: Path):
    train = frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)].copy()
    if train.empty:
        raise ValueError("No moneyline classifier training rows")
    if train["actual_home_win"].nunique() < 2:
        raise ValueError("Moneyline classifier target has only one class")
    model = build_pipeline()
    model.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["actual_home_win"].astype(int))
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "moneyline_classifier_v2_2.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "model_path": str(model_path),
        "training_rows": int(len(train)),
        "training_hash": canonical_hash(train),
        "target": "actual_home_win",
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "sklearn_version": sklearn.__version__,
        "research_version": RESEARCH_VERSION,
    }
    return model, metadata


def score_frame(frame: pd.DataFrame, model, cfg: MoneylineClassifierConfig) -> pd.DataFrame:
    games = frame.copy()
    games["home_win_probability"] = model.predict_proba(games[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    games["away_win_probability"] = 1.0 - games["home_win_probability"]
    home = games.copy()
    home["side"] = "HOME"
    home["display_side"] = home.get("home_team", "HOME")
    home["model_probability"] = home["home_win_probability"]
    home["price"] = home["home_moneyline"]
    home["implied_probability"] = home["home_moneyline_implied_probability"]
    home["no_vig_probability"] = home["home_moneyline_no_vig_probability"]
    home["actual_result"] = np.where(home["actual_home_win"], "WIN", "LOSS")

    away = games.copy()
    away["side"] = "AWAY"
    away["display_side"] = away.get("away_team", "AWAY")
    away["model_probability"] = away["away_win_probability"]
    away["price"] = away["away_moneyline"]
    away["implied_probability"] = away["away_moneyline_implied_probability"]
    away["no_vig_probability"] = away["away_moneyline_no_vig_probability"]
    away["actual_result"] = np.where(~away["actual_home_win"], "WIN", "LOSS")

    out = pd.concat([home, away], ignore_index=True, sort=False)
    out["payout_per_unit"] = american_profit_per_unit(out["price"])
    out["probability_edge"] = out["model_probability"] - out["no_vig_probability"]
    out["ev"] = out["model_probability"] * out["payout_per_unit"] - (1.0 - out["model_probability"])
    out["rank_score"] = (out["ev"] * 100.0 + out["model_probability"] + out["probability_edge"]).round(6)
    out["eligible"] = (
        out["model_probability"].ge(cfg.min_model_probability)
        & out["probability_edge"].ge(cfg.min_probability_edge)
        & out["payout_per_unit"].notna()
    )
    if cfg.require_positive_ev:
        out["eligible"] = out["eligible"] & out["ev"].ge(cfg.min_ev)
    out["execution_action"] = "BLOCK"
    out["block_reason"] = "ML_PROBABILITY_EV_GATE"
    selected = []
    test = out[out["season"].eq(cfg.test_season)].copy()
    best_by_game = []
    for _game_pk, group in test[test["eligible"]].groupby("game_pk", sort=False):
        best_by_game.append(
            group.sort_values(
                ["rank_score", "ev", "model_probability", "probability_edge", "side"],
                ascending=[False, False, False, False, True],
                kind="mergesort",
            ).head(1)
        )
    game_pool = pd.concat(best_by_game, ignore_index=False, sort=False) if best_by_game else pd.DataFrame()
    for _game_date, group in game_pool.groupby("game_date", sort=True):
        selected.append(
            group.sort_values(
                ["rank_score", "ev", "model_probability", "probability_edge", "game_pk", "side"],
                ascending=[False, False, False, False, True, True],
                kind="mergesort",
            ).head(cfg.max_daily_picks)
        )
    if selected:
        selected_idx = pd.concat(selected).index
        out.loc[selected_idx, "execution_action"] = "BET"
        out.loc[selected_idx, "block_reason"] = ""
    out["profit_units"] = 0.0
    bets = out["execution_action"].eq("BET")
    out.loc[bets & out["actual_result"].eq("WIN"), "profit_units"] = out.loc[bets & out["actual_result"].eq("WIN"), "payout_per_unit"]
    out.loc[bets & out["actual_result"].eq("LOSS"), "profit_units"] = -1.0
    return out.sort_values(["game_date", "execution_action", "rank_score"], ascending=[True, True, False]).reset_index(drop=True)


def summarize(scored: pd.DataFrame, cfg: MoneylineClassifierConfig) -> dict[str, pd.DataFrame]:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    bets = test[test["execution_action"].eq("BET")].copy()

    def row_for(group: pd.DataFrame, label: str) -> dict:
        wins = int(group["actual_result"].eq("WIN").sum())
        losses = int(group["actual_result"].eq("LOSS").sum())
        profit = float(group["profit_units"].sum()) if len(group) else 0.0
        decisions = wins + losses
        return {
            "segment": label,
            "bets": int(len(group)),
            "wins": wins,
            "losses": losses,
            "win_rate": float(wins / decisions) if decisions else 0.0,
            "profit_units": profit,
            "roi": float(profit / len(group)) if len(group) else 0.0,
            "avg_model_probability": float(group["model_probability"].mean()) if len(group) else 0.0,
            "avg_probability_edge": float(group["probability_edge"].mean()) if len(group) else 0.0,
            "avg_ev": float(group["ev"].mean()) if len(group) else 0.0,
        }

    rows = [row_for(bets, "ALL")]
    for side, group in bets.groupby("side", sort=True):
        rows.append(row_for(group, side))
    daily = []
    for game_date, group in bets.groupby("game_date", sort=True):
        daily.append({"game_date": game_date, **row_for(group, "ALL")})
    weekly = []
    if not bets.empty:
        dates = pd.to_datetime(bets["game_date"], errors="raise")
        bets["test_week"] = dates.dt.isocalendar().week.astype(int)
        bets["week_start"] = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.date.astype(str)
        for (week, week_start), group in bets.groupby(["test_week", "week_start"], sort=True):
            weekly.append({"test_week": int(week), "week_start": week_start, **row_for(group, "ALL")})
    return {
        "overall": pd.DataFrame(rows),
        "daily": pd.DataFrame(daily),
        "weekly": pd.DataFrame(weekly),
        "scored_candidates": scored,
        "orders": bets.reset_index(drop=True),
    }


def model_metrics(scored: pd.DataFrame, cfg: MoneylineClassifierConfig) -> pd.DataFrame:
    test_games = scored[scored["season"].eq(cfg.test_season)].drop_duplicates("game_pk").copy()
    y = test_games["actual_home_win"].astype(int)
    p = test_games["home_win_probability"]
    row = {
        "rows": int(len(test_games)),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "accuracy_at_50": accuracy_score(y, p >= 0.5),
    }
    if y.nunique() > 1:
        row["auc"] = roc_auc_score(y, p)
    return pd.DataFrame([row])


def run_research(predictions: pd.DataFrame, features: pd.DataFrame | None, cfg: MoneylineClassifierConfig, model_dir: Path) -> dict[str, pd.DataFrame]:
    frame = build_moneyline_frame(predictions, features)
    model, metadata = fit_classifier(frame, cfg, model_dir)
    scored = score_frame(frame, model, cfg)
    result = summarize(scored, cfg)
    result["model_metrics"] = model_metrics(scored, cfg)
    result["model_metadata"] = pd.DataFrame([metadata])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLB moneyline v2.2 research: calibrated classifier + EV-ranked daily picks.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--train-start-season", default=MoneylineClassifierConfig.train_start_season, type=int)
    parser.add_argument("--train-end-season", default=MoneylineClassifierConfig.train_end_season, type=int)
    parser.add_argument("--test-season", default=MoneylineClassifierConfig.test_season, type=int)
    parser.add_argument("--min-model-probability", default=MoneylineClassifierConfig.min_model_probability, type=float)
    parser.add_argument("--min-probability-edge", default=MoneylineClassifierConfig.min_probability_edge, type=float)
    parser.add_argument("--min-ev", default=MoneylineClassifierConfig.min_ev, type=float)
    parser.add_argument("--max-daily-picks", default=MoneylineClassifierConfig.max_daily_picks, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MoneylineClassifierConfig(
        train_start_season=args.train_start_season,
        train_end_season=args.train_end_season,
        test_season=args.test_season,
        min_model_probability=args.min_model_probability,
        min_probability_edge=args.min_probability_edge,
        min_ev=args.min_ev,
        max_daily_picks=args.max_daily_picks,
    )
    predictions = pd.read_csv(args.predictions)
    features = pd.read_csv(args.features) if args.features else None
    model_dir = args.model_dir or args.out_dir / "models"
    result = run_research(predictions, features, cfg, model_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "research_version": RESEARCH_VERSION,
            "generated_at_utc": utc_now_iso(),
            "predictions_hash": canonical_hash(predictions),
            "features_hash": canonical_hash(features) if features is not None else None,
            "config": asdict(cfg),
            "outputs": {name: canonical_hash(frame) for name, frame in result.items()},
        },
    )
    print(result["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
