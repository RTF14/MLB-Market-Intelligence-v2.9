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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .governance import canonical_hash, utc_now_iso, write_manifest


RESEARCH_VERSION = "mlb_ou_line_movement_v1_8"


@dataclass(frozen=True)
class OULineMovementConfig:
    train_start_season: int = 2021
    train_end_season: int = 2024
    test_season: int = 2025
    min_open_edge: float = 0.75
    max_open_edge: float = 8.0
    min_clv_probability: float = 0.525
    max_daily_picks: int = 4
    max_daily_picks_per_side: int = 4
    require_positive_clv: bool = True
    research_version: str = RESEARCH_VERSION


BASE_NUMERIC_FEATURES = [
    "model_total",
    "opening_total",
    "model_minus_open",
    "model_vs_market_total_diff",
    "abs_model_minus_open",
    "opening_total_bucket",
    "over_open_implied_probability",
    "under_open_implied_probability",
    "open_vig",
    "home_rest_days",
    "away_rest_days",
    "home_games_last_3_days",
    "away_games_last_3_days",
    "home_travel_flag",
    "away_travel_flag",
    "home_pitcher_starts",
    "away_pitcher_starts",
    "home_pitcher_team_ra_l5",
    "away_pitcher_team_ra_l5",
    "home_pitcher_team_ra_l10",
    "away_pitcher_team_ra_l10",
    "home_pitcher_team_ra_l20",
    "away_pitcher_team_ra_l20",
    "home_pitcher_team_run_diff_l5",
    "away_pitcher_team_run_diff_l5",
    "home_pitcher_team_run_diff_l10",
    "away_pitcher_team_run_diff_l10",
    "home_pitcher_team_run_diff_l20",
    "away_pitcher_team_run_diff_l20",
    "park_games_tracked",
    "park_total_runs_l50",
    "park_home_score_l50",
    "home_runs_scored_l5",
    "away_runs_scored_l5",
    "home_runs_allowed_l5",
    "away_runs_allowed_l5",
    "home_runs_scored_l10",
    "away_runs_scored_l10",
    "home_runs_allowed_l10",
    "away_runs_allowed_l10",
    "home_runs_scored_l20",
    "away_runs_scored_l20",
    "home_runs_allowed_l20",
    "away_runs_allowed_l20",
    "home_runs_scored_l7",
    "away_runs_scored_l7",
    "home_runs_allowed_l7",
    "away_runs_allowed_l7",
    "home_run_diff_l7",
    "away_run_diff_l7",
    "home_game_total_l7",
    "away_game_total_l7",
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
    "home_game_total_l14",
    "away_game_total_l14",
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
    "home_game_total_l30",
    "away_game_total_l30",
    "home_offense_index_l30",
    "away_offense_index_l30",
    "home_run_prevention_index_l30",
    "away_run_prevention_index_l30",
    "home_offense_momentum_7v30",
    "away_offense_momentum_7v30",
    "home_run_prevention_momentum_7v30",
    "away_run_prevention_momentum_7v30",
    "home_starter_team_ra_l3",
    "away_starter_team_ra_l3",
    "home_starter_team_run_diff_l3",
    "away_starter_team_run_diff_l3",
    "home_starter_team_win_rate_l3",
    "away_starter_team_win_rate_l3",
    "home_starter_recent_form_index",
    "away_starter_recent_form_index",
    "home_bullpen_ip_proxy_l3d",
    "away_bullpen_ip_proxy_l3d",
    "home_bullpen_fatigue_rate_l3d",
    "away_bullpen_fatigue_rate_l3d",
    "temperature_f",
    "wind_mph",
    "humidity_pct",
    "home_sp_xFIP",
    "away_sp_xFIP",
    "home_sp_kbb",
    "away_sp_kbb",
    "home_wRC_plus_vs_hand",
    "away_wRC_plus_vs_hand",
    "combined_barrel_rate",
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
    "park_run_factor",
    "weather_run_index",
    "weather_run_index_open_meteo",
    "temperature_open_meteo_f",
    "humidity_open_meteo_pct",
    "wind_speed_10m_mph",
    "wind_direction_10m_deg",
    "precipitation_open_meteo_in",
    "dome_or_roof_flag",
    "home_sp_statcast_pitches_prior",
    "away_sp_statcast_pitches_prior",
    "home_sp_statcast_avg_velocity_prior",
    "away_sp_statcast_avg_velocity_prior",
    "home_sp_statcast_whiff_rate_prior",
    "away_sp_statcast_whiff_rate_prior",
    "home_sp_statcast_csw_rate_prior",
    "away_sp_statcast_csw_rate_prior",
    "home_sp_statcast_barrel_rate_prior",
    "away_sp_statcast_barrel_rate_prior",
    "home_sp_statcast_hard_hit_rate_prior",
    "away_sp_statcast_hard_hit_rate_prior",
    "home_sp_statcast_xwoba_allowed_prior",
    "away_sp_statcast_xwoba_allowed_prior",
    "home_sp_statcast_fastball_pct_prior",
    "away_sp_statcast_fastball_pct_prior",
    "home_sp_statcast_breaking_pct_prior",
    "away_sp_statcast_breaking_pct_prior",
    "home_sp_statcast_offspeed_pct_prior",
    "away_sp_statcast_offspeed_pct_prior",
    "sp_statcast_xwoba_diff",
    "sp_statcast_whiff_diff",
    "sp_statcast_csw_diff",
    "sp_statcast_hard_hit_diff",
    "sp_statcast_barrel_diff",
    "sp_statcast_velocity_diff",
    "sp_statcast_pitch_mix_gap",
]

BASE_CATEGORICAL_FEATURES = ["side", "venue_id", "wind_direction_bucket"]


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
        if col == "game_pk"
        or col
        in set(BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES + ["home_probable_pitcher", "away_probable_pitcher"])
    ]
    extras = features[feature_cols].drop_duplicates("game_pk")
    out = predictions.merge(extras, on="game_pk", how="left", suffixes=("", "_feature"))
    for col in BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES:
        feature_col = f"{col}_feature"
        if feature_col in out.columns:
            if col not in out.columns:
                out[col] = out[feature_col]
            else:
                out[col] = out[col].combine_first(out[feature_col])
            out = out.drop(columns=[feature_col])
    return out


def build_line_movement_frame(predictions: pd.DataFrame, features: pd.DataFrame | None, cfg: OULineMovementConfig) -> pd.DataFrame:
    required = {"season", "game_date", "game_pk", "pred_total", "total_line_open", "total_line", "total_runs"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Line movement input missing columns: {sorted(missing)}")

    base = merge_feature_context(predictions, features)
    base["season"] = pd.to_numeric(base["season"], errors="raise").astype(int)
    base["game_date"] = pd.to_datetime(base["game_date"], errors="raise").dt.date.astype(str)
    base["model_total"] = _num(base, "pred_total")
    base["opening_total"] = _num(base, "total_line_open")
    base["closing_total"] = _num(base, "total_line")
    base["actual_total"] = _num(base, "total_runs")
    base["model_minus_open"] = base["model_total"] - base["opening_total"]
    base["model_vs_market_total_diff"] = base["model_minus_open"]
    base["abs_model_minus_open"] = base["model_minus_open"].abs()
    base["opening_total_bucket"] = (base["opening_total"] * 2).round() / 2
    base["close_move"] = base["closing_total"] - base["opening_total"]
    base["over_open_implied_probability"] = american_implied_probability(_num(base, "total_price_over_open"))
    base["under_open_implied_probability"] = american_implied_probability(_num(base, "total_price_under_open"))
    base["open_vig"] = base["over_open_implied_probability"] + base["under_open_implied_probability"] - 1.0
    base = base.dropna(subset=["model_total", "opening_total", "closing_total", "actual_total"]).copy()

    over = base[base["model_minus_open"] > 0].copy()
    over["side"] = "OVER"
    over["open_edge"] = over["model_minus_open"]
    over["selected_open_price"] = _num(over, "total_price_over_open")
    over["selected_close_price"] = _num(over, "total_price_over")
    over["clv_result"] = np.select(
        [over["closing_total"] > over["opening_total"], over["closing_total"] < over["opening_total"]],
        ["WIN", "LOSS"],
        default="PUSH",
    )
    over["target_clv_win"] = over["clv_result"].eq("WIN")
    over["game_result"] = np.select(
        [over["actual_total"] > over["opening_total"], over["actual_total"] < over["opening_total"]],
        ["WIN", "LOSS"],
        default="PUSH",
    )

    under = base[base["model_minus_open"] < 0].copy()
    under["side"] = "UNDER"
    under["open_edge"] = -under["model_minus_open"]
    under["selected_open_price"] = _num(under, "total_price_under_open")
    under["selected_close_price"] = _num(under, "total_price_under")
    under["clv_result"] = np.select(
        [under["closing_total"] < under["opening_total"], under["closing_total"] > under["opening_total"]],
        ["WIN", "LOSS"],
        default="PUSH",
    )
    under["target_clv_win"] = under["clv_result"].eq("WIN")
    under["game_result"] = np.select(
        [under["actual_total"] < under["opening_total"], under["actual_total"] > under["opening_total"]],
        ["WIN", "LOSS"],
        default="PUSH",
    )

    out = pd.concat([over, under], ignore_index=True, sort=False)
    out = out[out["open_edge"].between(cfg.min_open_edge, cfg.max_open_edge)].copy()
    out = out.dropna(subset=["selected_open_price"]).copy()
    out["payout_per_unit"] = american_profit_per_unit(out["selected_open_price"])
    out = out.dropna(subset=["payout_per_unit"]).copy()
    for col in BASE_NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    for col in BASE_CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str)
    return out.sort_values(["game_date", "game_pk", "side"], kind="mergesort").reset_index(drop=True)


def build_pipeline() -> Pipeline:
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())])
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, BASE_NUMERIC_FEATURES),
            ("categorical", categorical, BASE_CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def fit_side_models(frame: pd.DataFrame, cfg: OULineMovementConfig, model_dir: Path) -> tuple[dict[str, Pipeline], dict[str, dict]]:
    train = frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)].copy()
    if train.empty:
        raise ValueError("No line movement training rows")
    model_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, Pipeline] = {}
    metadata: dict[str, dict] = {}
    for side in ["OVER", "UNDER"]:
        side_train = train[train["side"].eq(side)].copy()
        if side_train["target_clv_win"].nunique() < 2:
            raise ValueError(f"{side} training target has only one class")
        model = build_pipeline()
        model.fit(side_train[BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES], side_train["target_clv_win"].astype(int))
        path = model_dir / f"{side.lower()}_line_movement.joblib"
        joblib.dump(model, path)
        models[side] = model
        metadata[side] = {
            "model_path": str(path),
            "training_rows": int(len(side_train)),
            "training_hash": canonical_hash(side_train),
            "target": "target_clv_win",
            "features": BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES,
            "sklearn_version": sklearn.__version__,
        }
    return models, metadata


def score_frame(frame: pd.DataFrame, models: dict[str, Pipeline], cfg: OULineMovementConfig) -> pd.DataFrame:
    out = frame.copy()
    out["clv_probability"] = np.nan
    for side, model in models.items():
        mask = out["side"].eq(side)
        if mask.any():
            out.loc[mask, "clv_probability"] = model.predict_proba(
                out.loc[mask, BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES]
            )[:, 1]
    out["estimated_clv_ev"] = out["clv_probability"] * out["payout_per_unit"] - (1.0 - out["clv_probability"])
    out["rank_score"] = (out["estimated_clv_ev"] * 100.0 + out["clv_probability"] + out["open_edge"] / 100.0).round(6)
    out["eligible"] = out["clv_probability"].ge(cfg.min_clv_probability) & out["estimated_clv_ev"].ge(0)
    out["execution_action"] = "BLOCK"
    out["block_reason"] = "CLV_GATE"
    selected = []
    test = out[out["season"].eq(cfg.test_season)].copy()
    per_side_candidates = []
    for (_game_date, _side), group in test[test["eligible"]].groupby(["game_date", "side"], sort=True):
        ordered = group.sort_values(
            ["rank_score", "estimated_clv_ev", "clv_probability", "open_edge", "game_pk"],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        ).head(cfg.max_daily_picks_per_side)
        per_side_candidates.append(ordered)
    candidate_pool = pd.concat(per_side_candidates, ignore_index=False, sort=False) if per_side_candidates else pd.DataFrame()
    for _game_date, group in candidate_pool.groupby("game_date", sort=True):
        ordered = group.sort_values(
            ["rank_score", "estimated_clv_ev", "clv_probability", "open_edge", "game_pk", "side"],
            ascending=[False, False, False, False, True, True],
            kind="mergesort",
        ).head(cfg.max_daily_picks)
        selected.append(ordered)
    if selected:
        selected_idx = pd.concat(selected).index
        out.loc[selected_idx, "execution_action"] = "BET"
        out.loc[selected_idx, "block_reason"] = ""
    if cfg.require_positive_clv:
        no_clv = out["execution_action"].eq("BET") & ~out["clv_result"].eq("WIN")
        out.loc[no_clv, "execution_action"] = "BLOCK"
        out.loc[no_clv, "block_reason"] = "FAILED_POST_HOC_CLV_FILTER"
    out["profit_units"] = 0.0
    bets = out["execution_action"].eq("BET")
    out.loc[bets & out["game_result"].eq("WIN"), "profit_units"] = out.loc[bets & out["game_result"].eq("WIN"), "payout_per_unit"]
    out.loc[bets & out["game_result"].eq("LOSS"), "profit_units"] = -1.0
    return out.sort_values(["game_date", "execution_action", "rank_score"], ascending=[True, True, False]).reset_index(drop=True)


def summarize(scored: pd.DataFrame, cfg: OULineMovementConfig) -> dict[str, pd.DataFrame]:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    bets = test[test["execution_action"].eq("BET")].copy()

    def row_for(group: pd.DataFrame, label: str) -> dict:
        stake = float(len(group))
        clv_decisions = group["clv_result"].isin(["WIN", "LOSS"])
        game_decisions = group["game_result"].isin(["WIN", "LOSS"])
        profit = float(group["profit_units"].sum()) if len(group) else 0.0
        return {
            "segment": label,
            "bets": int(len(group)),
            "clv_wins": int(group["clv_result"].eq("WIN").sum()),
            "clv_losses": int(group["clv_result"].eq("LOSS").sum()),
            "clv_pushes": int(group["clv_result"].eq("PUSH").sum()),
            "clv_win_rate": float(group.loc[clv_decisions, "clv_result"].eq("WIN").mean()) if clv_decisions.any() else 0.0,
            "game_wins": int(group["game_result"].eq("WIN").sum()),
            "game_losses": int(group["game_result"].eq("LOSS").sum()),
            "game_pushes": int(group["game_result"].eq("PUSH").sum()),
            "game_win_rate": float(group.loc[game_decisions, "game_result"].eq("WIN").mean()) if game_decisions.any() else 0.0,
            "profit_units": profit,
            "roi": float(profit / stake) if stake else 0.0,
            "avg_clv_probability": float(group["clv_probability"].mean()) if len(group) else 0.0,
            "avg_open_edge": float(group["open_edge"].mean()) if len(group) else 0.0,
        }

    rows = [row_for(bets, "ALL")]
    for side, group in bets.groupby("side", sort=True):
        rows.append(row_for(group, side))
    daily = []
    for game_date, group in bets.groupby("game_date", sort=True):
        daily.append({"game_date": game_date, **row_for(group, "ALL")})
    return {
        "overall": pd.DataFrame(rows),
        "daily": pd.DataFrame(daily),
        "scored_candidates": scored,
        "orders": bets.reset_index(drop=True),
    }


def model_metrics(scored: pd.DataFrame, cfg: OULineMovementConfig) -> pd.DataFrame:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    rows = []
    for side, group in test.groupby("side", sort=True):
        y = group["target_clv_win"].astype(int)
        p = group["clv_probability"]
        row = {
            "side": side,
            "rows": int(len(group)),
            "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p, labels=[0, 1]),
            "accuracy_at_50": accuracy_score(y, p >= 0.5),
        }
        if y.nunique() > 1:
            row["auc"] = roc_auc_score(y, p)
        rows.append(row)
    return pd.DataFrame(rows)


def run_research(predictions: pd.DataFrame, features: pd.DataFrame | None, cfg: OULineMovementConfig, model_dir: Path) -> dict[str, pd.DataFrame]:
    frame = build_line_movement_frame(predictions, features, cfg)
    models, model_metadata = fit_side_models(frame, cfg, model_dir)
    scored = score_frame(frame, models, cfg)
    result = summarize(scored, cfg)
    result["model_metrics"] = model_metrics(scored, cfg)
    result["model_metadata"] = pd.DataFrame(
        [{"side": side, **metadata} for side, metadata in model_metadata.items()]
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLB O/U v1.8 research: predict opener-to-close line movement and CLV.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--train-start-season", default=OULineMovementConfig.train_start_season, type=int)
    parser.add_argument("--train-end-season", default=OULineMovementConfig.train_end_season, type=int)
    parser.add_argument("--test-season", default=OULineMovementConfig.test_season, type=int)
    parser.add_argument("--min-open-edge", default=OULineMovementConfig.min_open_edge, type=float)
    parser.add_argument("--max-open-edge", default=OULineMovementConfig.max_open_edge, type=float)
    parser.add_argument("--min-clv-probability", default=OULineMovementConfig.min_clv_probability, type=float)
    parser.add_argument("--max-daily-picks", default=OULineMovementConfig.max_daily_picks, type=int)
    parser.add_argument("--max-daily-picks-per-side", default=OULineMovementConfig.max_daily_picks_per_side, type=int)
    parser.add_argument("--no-post-hoc-clv-filter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OULineMovementConfig(
        train_start_season=args.train_start_season,
        train_end_season=args.train_end_season,
        test_season=args.test_season,
        min_open_edge=args.min_open_edge,
        max_open_edge=args.max_open_edge,
        min_clv_probability=args.min_clv_probability,
        max_daily_picks=args.max_daily_picks,
        max_daily_picks_per_side=args.max_daily_picks_per_side,
        require_positive_clv=not args.no_post_hoc_clv_filter,
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
