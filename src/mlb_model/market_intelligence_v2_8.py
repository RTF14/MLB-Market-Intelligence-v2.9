from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .governance import canonical_hash, utc_now_iso, write_manifest
from .market_intelligence_v2_5 import _num, _summary_row, build_pipeline
from .market_intelligence_v2_6 import INJURY_NUMERIC_FEATURES, ML_NUMERIC_FEATURES_V26, OU_INTERACTION_FEATURES_V26, enrich_ou_frame_v26
from .market_intelligence_v2_7 import (
    ML_CATEGORICAL_FEATURES_V27,
    MarketIntelligenceV27Config,
    _allowed_ml_segment_mask,
    _ml_model_metrics,
    _ml_side_segment,
    _ml_thresholds,
    _ou_model_metrics,
    _shrunk_values,
    _total_bucket_group,
    _training_frame,
    fit_group_models_v27,
    prepare_ml_frame_v27,
    run_ou_v27,
)
from .market_snapshots import add_clv_snapshot_fields, build_snapshot_tables
from .ou_line_movement_model import BASE_CATEGORICAL_FEATURES, BASE_NUMERIC_FEATURES, build_line_movement_frame


RESEARCH_VERSION = "mlb_market_intelligence_v2_8"


@dataclass(frozen=True)
class MarketIntelligenceV28Config(MarketIntelligenceV27Config):
    ml_uncertainty_penalty: float = 0.65
    ou_uncertainty_penalty: float = 0.60
    min_segment_stake_multiplier: float = 0.25
    max_segment_stake_multiplier: float = 1.50
    max_daily_pitcher_exposure: int = 1
    max_daily_weather_cluster_exposure: int = 2
    research_version: str = RESEARCH_VERSION


ML_EXTRA_FEATURES_V28 = [
    "ml_prediction_variance",
    "ml_pregame_uncertainty",
    "ml_lineup_value_removed_for_side",
    "ml_opponent_lineup_value_removed",
    "ml_lineup_strength_diff",
    "ml_top3_hitters_missing_flag",
    "ml_pitcher_volatility_for_side",
    "ml_bullpen_volatility_for_side",
    "ml_model_wrong_rate_l5",
    "ml_market_loss_rate_l5",
    "ml_team_overperformance_l5",
    "ml_edge_x_uncertainty",
    "ml_lineup_x_edge",
]

ML_CATEGORICAL_FEATURES_V28 = list(dict.fromkeys(ML_CATEGORICAL_FEATURES_V27 + ["recommended_timing", "time_bucket"]))
ML_NUMERIC_FEATURES_V28 = list(dict.fromkeys(ML_NUMERIC_FEATURES_V26 + ML_EXTRA_FEATURES_V28))

OU_EXTRA_FEATURES_V28 = [
    "ou_prediction_variance",
    "ou_team_scoring_variance",
    "ou_pitcher_volatility",
    "ou_bullpen_volatility",
    "ou_model_disagreement",
    "ou_uncertainty",
    "ou_lineup_run_value_removed",
    "ou_top3_hitters_missing_flag",
    "wind_out_flag",
    "wind_in_flag",
    "dome_flag",
    "bullpen_ip_l1_l3_diff",
    "bullpen_ip_l3_l5_diff",
    "ou_model_wrong_rate_l5",
    "ou_market_loss_rate_l5",
]

OU_CATEGORICAL_FEATURES_V28 = list(dict.fromkeys(BASE_CATEGORICAL_FEATURES + ["total_bucket_group", "recommended_timing", "weather_cluster", "wind_direction_bucket"]))


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(0.0, 1.0)


def _rolling_shifted_rate(df: pd.DataFrame, group_cols: list[str], value_col: str, window: int = 5) -> pd.Series:
    out = pd.Series(0.0, index=df.index, dtype=float)
    if value_col not in df.columns:
        return out
    work = df[group_cols + [value_col]].copy()
    work["_idx"] = df.index
    for _, group in work.groupby(group_cols, sort=False):
        vals = pd.to_numeric(group[value_col], errors="coerce").fillna(0.0)
        out.loc[group["_idx"]] = vals.shift(1).rolling(window, min_periods=1).mean().fillna(0.0).to_numpy()
    return out


def _lineup_removed(home_bat: pd.Series, away_bat: pd.Series, home_sev: pd.Series, away_sev: pd.Series, side: pd.Series) -> tuple[pd.Series, pd.Series]:
    home_removed = 0.18 * pd.to_numeric(home_bat, errors="coerce").fillna(0.0) + 0.04 * pd.to_numeric(home_sev, errors="coerce").fillna(0.0)
    away_removed = 0.18 * pd.to_numeric(away_bat, errors="coerce").fillna(0.0) + 0.04 * pd.to_numeric(away_sev, errors="coerce").fillna(0.0)
    selected = pd.Series(np.where(side.astype(str).eq("HOME"), home_removed, away_removed), index=side.index)
    opponent = pd.Series(np.where(side.astype(str).eq("HOME"), away_removed, home_removed), index=side.index)
    return selected, opponent


def prepare_ml_frame_v28(scored: pd.DataFrame, cfg: MarketIntelligenceV28Config) -> pd.DataFrame:
    out = prepare_ml_frame_v27(scored, cfg)
    selected_removed, opponent_removed = _lineup_removed(
        _num(out, "home_bat_injury_count"),
        _num(out, "away_bat_injury_count"),
        _num(out, "home_injury_severity_sum"),
        _num(out, "away_injury_severity_sum"),
        out["side"],
    )
    out["ml_lineup_value_removed_for_side"] = selected_removed
    out["ml_opponent_lineup_value_removed"] = opponent_removed
    out["ml_lineup_strength_diff"] = opponent_removed - selected_removed
    out["ml_top3_hitters_missing_flag"] = (selected_removed >= 0.60).astype(int)
    out["ml_pitcher_volatility_for_side"] = (
        _num(out, "starter_recent_form_diff_for_side", 0.0).abs()
        + 0.35 * _num(out, "starter_xFIP_diff_for_side", 0.0).abs()
        + 0.04 * _num(out, "starter_kbb_diff_for_side", 0.0).abs()
    )
    out["ml_bullpen_volatility_for_side"] = (
        _num(out, "bullpen_fatigue_diff_for_side", 0.0).abs()
        + 0.25 * _num(out, "bullpen_xFIP_diff_for_side", 0.0).abs()
        + 0.15 * _num(out, "bullpen_fatigue_index", 0.0).abs()
    )
    out["time_bucket"] = out.get("time_bucket", pd.Series("open", index=out.index)).fillna("open").astype(str)
    strong_edge = out["probability_edge"].ge(0.08) & out["ev_open"].ge(0.06)
    steam_agrees = _num(out, "pregame_steam_direction", 0.0).ge(0)
    out["recommended_timing"] = np.select([strong_edge & steam_agrees, strong_edge & ~steam_agrees], ["BET_EARLY", "WAIT_FOR_CONFIRMATION"], default="MID_OR_CLOSE")
    out["ml_model_wrong"] = np.where(out["actual_result"].eq("WIN"), out["model_probability"].lt(0.5), out["model_probability"].ge(0.5)).astype(int)
    out["ml_market_loss"] = out["clv_result"].eq("LOSS").astype(int)
    out["ml_overperformance"] = out["target_game_win"] - out["model_probability"].fillna(0.5)
    out = out.sort_values(["game_date", "game_pk", "side"], kind="mergesort").reset_index(drop=True)
    out["ml_model_wrong_rate_l5"] = _rolling_shifted_rate(out, ["ml_side_segment"], "ml_model_wrong")
    out["ml_market_loss_rate_l5"] = _rolling_shifted_rate(out, ["ml_side_segment"], "ml_market_loss")
    out["ml_team_overperformance_l5"] = _rolling_shifted_rate(out, ["display_side"], "ml_overperformance") if "display_side" in out.columns else 0.0
    out["ml_pregame_uncertainty"] = (
        _num(out, "pregame_implied_probability_move", 0.0).abs()
        + 0.015 * _num(out, "ml_pitcher_volatility_for_side", 0.0)
        + 0.010 * _num(out, "ml_bullpen_volatility_for_side", 0.0)
        + 0.05 * _num(out, "pregame_stale_price_flag", 0.0)
    ).fillna(0.0).clip(0, 0.75)
    model_prob = _num(out, "model_probability", 0.5).fillna(0.5)
    out["ml_prediction_variance"] = (model_prob * (1.0 - model_prob) + out["ml_pregame_uncertainty"]).fillna(0.25).clip(0.001, 1.0)
    out["ml_edge_x_uncertainty"] = out["probability_edge"] * out["ml_pregame_uncertainty"]
    out["ml_lineup_x_edge"] = out["ml_lineup_strength_diff"] * out["probability_edge"]
    for col in ML_EXTRA_FEATURES_V28:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ML_CATEGORICAL_FEATURES_V28:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str)
    return out


def fit_group_models_v28(frame: pd.DataFrame, cfg: MarketIntelligenceV28Config, target: str) -> tuple[dict[str, object], pd.DataFrame]:
    train = _training_frame(frame, cfg)
    models = {}
    rows = []
    for group_name, group in train.groupby("favorite_group", sort=True):
        if group[target].nunique() < 2:
            continue
        model = build_pipeline(ML_NUMERIC_FEATURES_V28, ML_CATEGORICAL_FEATURES_V28)
        model.fit(group[ML_NUMERIC_FEATURES_V28 + ML_CATEGORICAL_FEATURES_V28], group[target].astype(int))
        models[group_name] = model
        rows.append({"market": "ML", "model_group": group_name, "target": target, "training_rows": int(len(group)), "oof_training_rows": int(group["base_prediction_oof"].sum()), "training_hash": canonical_hash(group), "sklearn_version": sklearn.__version__})
    if not models:
        raise ValueError(f"No ML models trained for {target}")
    return models, pd.DataFrame(rows)


def score_ml_v28(frame: pd.DataFrame, clv_models: dict[str, object], game_models: dict[str, object], cfg: MarketIntelligenceV28Config) -> pd.DataFrame:
    out = frame.copy()
    out["ml_clv_probability"] = np.nan
    out["ml_game_probability"] = np.nan
    for group_name, model in clv_models.items():
        mask = out["favorite_group"].eq(group_name)
        if mask.any():
            out.loc[mask, "ml_clv_probability"] = model.predict_proba(out.loc[mask, ML_NUMERIC_FEATURES_V28 + ML_CATEGORICAL_FEATURES_V28])[:, 1]
    for group_name, model in game_models.items():
        mask = out["favorite_group"].eq(group_name)
        if mask.any():
            out.loc[mask, "ml_game_probability"] = model.predict_proba(out.loc[mask, ML_NUMERIC_FEATURES_V28 + ML_CATEGORICAL_FEATURES_V28])[:, 1]
    out["ml_model_disagreement"] = (out["ml_game_probability"] - out["ml_clv_probability"]).abs().fillna(0.0)
    out["ml_total_uncertainty"] = (out["ml_prediction_variance"] + out["ml_model_disagreement"]).clip(0.001, 1.0)
    out["ml_clv_ev"] = out["ml_clv_probability"] * out["payout_per_unit"] - (1.0 - out["ml_clv_probability"])
    out["ml_game_ev_open"] = out["ml_game_probability"] * out["payout_per_unit"] - (1.0 - out["ml_game_probability"])
    out["ml_blended_ev"] = 0.60 * out["ml_game_ev_open"] + 0.40 * out["ml_clv_ev"]
    out["ml_uncertainty_penalty"] = (1.0 - cfg.ml_uncertainty_penalty * out["ml_total_uncertainty"]).clip(0.05, 1.0)
    out["ml_risk_adjusted_ev"] = out["ml_blended_ev"] * out["ml_uncertainty_penalty"]
    out["rank_score"] = (out["ml_risk_adjusted_ev"] * (out["ml_game_probability"] - 0.5).clip(lower=0.001) / np.sqrt(out["ml_total_uncertainty"])).round(6)
    return out


def _segment_stake_multiplier(shrunk_clv: float, shrunk_roi: float, cfg: MarketIntelligenceV28Config) -> float:
    raw = 1.0 + 3.0 * (shrunk_clv - 0.52) + 5.0 * shrunk_roi
    return float(np.clip(raw, cfg.min_segment_stake_multiplier, cfg.max_segment_stake_multiplier))


def build_ml_segment_gate_v28(scored: pd.DataFrame, cfg: MarketIntelligenceV28Config) -> pd.DataFrame:
    from .market_intelligence_v2_7 import _ml_thresholds

    train = _training_frame(scored, cfg)
    edge_threshold, ev_threshold = _ml_thresholds(train, cfg)
    train = train[
        train["model_probability"].ge(cfg.ml_min_win_probability)
        & train["ml_game_probability"].ge(cfg.ml_min_game_probability)
        & train["probability_edge"].ge(edge_threshold)
        & train["ev_open"].ge(ev_threshold)
        & train["ml_clv_probability"].ge(cfg.ml_min_clv_probability)
        & train["ml_game_ev_open"].ge(0)
        & train["ml_clv_ev"].ge(0)
    ].copy()
    rows = []
    for (segment, band), group in train.groupby(["ml_side_segment", "price_band"], sort=True):
        row = _summary_row(group, f"{segment}_{band}")
        shrunk_clv, shrunk_roi = _shrunk_values(group, cfg.ml_segment_prior_bets, cfg.ml_segment_prior_clv)
        row.update({"ml_side_segment": segment, "price_band": band, "shrunk_clv_win_rate": shrunk_clv, "shrunk_roi": shrunk_roi, "stake_multiplier": _segment_stake_multiplier(shrunk_clv, shrunk_roi, cfg)})
        row["allowed_segment"] = row["bets"] >= cfg.segment_min_bets and shrunk_clv >= cfg.ml_segment_min_shrunk_clv and shrunk_roi >= cfg.ml_segment_min_shrunk_roi and band != "FAV_220_PLUS"
        rows.append(row)
    return pd.DataFrame(rows)


def _merge_stake_multiplier(orders: pd.DataFrame, gate: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = orders.copy()
    if out.empty or gate.empty:
        out["stake_multiplier"] = 0.0 if out.empty else 1.0
        return out
    lookup = gate[keys + ["stake_multiplier"]].drop_duplicates(keys)
    out = out.merge(lookup, on=keys, how="left")
    out["stake_multiplier"] = pd.to_numeric(out["stake_multiplier"], errors="coerce").fillna(1.0)
    return out


def _pitcher_key(row: pd.Series) -> str:
    side = str(row.get("side", ""))
    if side == "HOME":
        return str(row.get("home_probable_pitcher", row.get("home_pitcher_name", "")))
    if side == "AWAY":
        return str(row.get("away_probable_pitcher", row.get("away_pitcher_name", "")))
    return ""


def _weather_cluster(row: pd.Series) -> str:
    venue = str(row.get("venue_id", ""))
    wind = pd.to_numeric(pd.Series([row.get("wind_mph", np.nan)]), errors="coerce").iloc[0]
    wind_bucket = "WIND_UNKNOWN" if pd.isna(wind) else "WIND_HIGH" if wind >= 12 else "WIND_MED" if wind >= 6 else "WIND_LOW"
    return f"{venue}_{wind_bucket}"


def select_ml_orders_v28(scored: pd.DataFrame, gate: pd.DataFrame, cfg: MarketIntelligenceV28Config) -> pd.DataFrame:
    from .market_intelligence_v2_7 import _ml_thresholds, _allowed_ml_segment_mask

    test = scored[scored["season"].eq(cfg.test_season)].copy()
    edge_threshold, ev_threshold = _ml_thresholds(test, cfg)
    eligible = test["model_probability"].ge(cfg.ml_min_win_probability) & test["ml_game_probability"].ge(cfg.ml_min_game_probability) & test["probability_edge"].ge(edge_threshold) & test["ev_open"].ge(ev_threshold) & test["ml_clv_probability"].ge(cfg.ml_min_clv_probability) & test["ml_game_ev_open"].ge(0) & test["ml_clv_ev"].ge(0) & _allowed_ml_segment_mask(test, gate)
    pool = test[eligible].copy()
    if pool.empty:
        pool["execution_action"] = []
        return pool
    pool["pitcher_exposure_key"] = pool.apply(_pitcher_key, axis=1)
    game_best = [group.sort_values(["rank_score", "ml_risk_adjusted_ev", "game_pk", "side"], ascending=[False, False, True, True]).head(1) for _, group in pool.groupby("game_pk", sort=False)]
    pool = pd.concat(game_best, ignore_index=False, sort=False)
    selected = []
    for _, group in pool.groupby("game_date", sort=True):
        used_teams: dict[str, int] = {}
        used_pitchers: dict[str, int] = {}
        picks = []
        ordered = group.sort_values(["rank_score", "ml_risk_adjusted_ev", "game_pk"], ascending=[False, False, True])
        for idx, row in ordered.iterrows():
            teams = [str(row.get("home_team", "")), str(row.get("away_team", ""))]
            pitcher = str(row.get("pitcher_exposure_key", ""))
            if any(used_teams.get(team, 0) >= cfg.max_daily_team_exposure for team in teams if team):
                continue
            if pitcher and pitcher.lower() != "nan" and used_pitchers.get(pitcher, 0) >= cfg.max_daily_pitcher_exposure:
                continue
            picks.append(idx)
            for team in teams:
                if team:
                    used_teams[team] = used_teams.get(team, 0) + 1
            if pitcher and pitcher.lower() != "nan":
                used_pitchers[pitcher] = used_pitchers.get(pitcher, 0) + 1
            if len(picks) >= cfg.ml_max_daily:
                break
        selected.append(ordered.loc[picks])
    orders = pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame(columns=test.columns)
    orders = _merge_stake_multiplier(orders, gate, ["ml_side_segment", "price_band"])
    orders = orders.sort_values(["game_date", "rank_score", "game_pk"], ascending=[True, False, True]).reset_index(drop=True)
    orders["execution_action"] = "BET"
    orders["market"] = "ML"
    return add_clv_snapshot_fields(orders, market="ML", mode=cfg.snapshot_mode)


def run_ml_v28(ml_scored_candidates: pd.DataFrame, cfg: MarketIntelligenceV28Config) -> dict[str, pd.DataFrame]:
    frame = prepare_ml_frame_v28(ml_scored_candidates, cfg)
    clv_models, clv_metadata = fit_group_models_v28(frame, cfg, "target_clv_win")
    game_models, game_metadata = fit_group_models_v28(frame, cfg, "target_game_win")
    scored = score_ml_v28(frame, clv_models, game_models, cfg)
    gate = build_ml_segment_gate_v28(scored, cfg)
    orders = select_ml_orders_v28(scored, gate, cfg)
    bet_snapshots, closing_snapshots = build_snapshot_tables(orders)
    return {
        "ml_scored_candidates": scored,
        "ml_orders": orders.reset_index(drop=True),
        "ml_bet_snapshots": bet_snapshots,
        "ml_closing_snapshots": closing_snapshots,
        "ml_overall": pd.DataFrame([_summary_row(orders, "BET")]),
        "ml_segment_gate": gate,
        "ml_model_metrics": pd.concat([_ml_model_metrics(scored, cfg, "ml_clv_probability", "target_clv_win", "clv"), _ml_model_metrics(scored, cfg, "ml_game_probability", "target_game_win", "game")], ignore_index=True),
        "ml_model_metadata": pd.concat([clv_metadata, game_metadata], ignore_index=True),
    }


def _wind_direction_bucket(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.lower()
    return pd.Series(np.select([text.str.contains("out"), text.str.contains("in"), text.str.contains("left|right|center")], ["OUT", "IN", "CROSS"], default="UNKNOWN"), index=series.index)


def enrich_ou_frame_v28(frame: pd.DataFrame) -> pd.DataFrame:
    out = enrich_ou_frame_v26(frame)
    out["target_game_win"] = out["game_result"].eq("WIN").astype(int)
    out["total_bucket_group"] = pd.Series(np.select([_num(out, "opening_total") <= 7.5, _num(out, "opening_total") >= 9.5], ["LOW", "HIGH"], default="NORMAL"), index=out.index)
    out["wind_direction_bucket"] = _wind_direction_bucket(out["wind_direction"] if "wind_direction" in out.columns else pd.Series("", index=out.index))
    out["wind_out_flag"] = out["wind_direction_bucket"].eq("OUT").astype(int)
    out["wind_in_flag"] = out["wind_direction_bucket"].eq("IN").astype(int)
    out["dome_flag"] = out.get("weather_condition", pd.Series("", index=out.index)).fillna("").astype(str).str.lower().str.contains("dome|roof").astype(int)
    home_removed = 0.18 * _num(out, "home_bat_injury_count", 0.0) + 0.04 * _num(out, "home_injury_severity_sum", 0.0)
    away_removed = 0.18 * _num(out, "away_bat_injury_count", 0.0) + 0.04 * _num(out, "away_injury_severity_sum", 0.0)
    out["ou_lineup_run_value_removed"] = home_removed + away_removed
    out["ou_top3_hitters_missing_flag"] = (out["ou_lineup_run_value_removed"] >= 1.2).astype(int)
    out["ou_team_scoring_variance"] = (_num(out, "home_game_total_l7", 0.0) - _num(out, "home_game_total_l30", 0.0)).abs() + (_num(out, "away_game_total_l7", 0.0) - _num(out, "away_game_total_l30", 0.0)).abs()
    out["ou_pitcher_volatility"] = (_num(out, "home_starter_team_ra_l3", 0.0) - _num(out, "home_pitcher_team_ra_l20", 0.0)).abs() + (_num(out, "away_starter_team_ra_l3", 0.0) - _num(out, "away_pitcher_team_ra_l20", 0.0)).abs()
    out["ou_bullpen_volatility"] = (_num(out, "home_bullpen_ip_proxy_l1d", 0.0) + _num(out, "away_bullpen_ip_proxy_l1d", 0.0) - 0.5 * (_num(out, "home_bullpen_ip_proxy_l3d", 0.0) + _num(out, "away_bullpen_ip_proxy_l3d", 0.0))).abs()
    out["bullpen_ip_l1_l3_diff"] = (_num(out, "home_bullpen_ip_proxy_l1d", 0.0) + _num(out, "away_bullpen_ip_proxy_l1d", 0.0)) - (_num(out, "home_bullpen_ip_proxy_l3d", 0.0) + _num(out, "away_bullpen_ip_proxy_l3d", 0.0)) / 3.0
    out["bullpen_ip_l3_l5_diff"] = (_num(out, "home_bullpen_ip_proxy_l3d", 0.0) + _num(out, "away_bullpen_ip_proxy_l3d", 0.0)) / 3.0 - (_num(out, "home_bullpen_ip_proxy_l5d", 0.0) + _num(out, "away_bullpen_ip_proxy_l5d", 0.0)) / 5.0
    out["ou_prediction_variance"] = (2.25 + 0.35 * out["ou_team_scoring_variance"] + 0.20 * out["ou_pitcher_volatility"] + 0.25 * out["ou_bullpen_volatility"] + 0.10 * _num(out, "wind_mph", 0.0).abs()).clip(1.0, 30.0)
    out["ou_result_probability_formula"] = 0.5 + (_num(out, "open_edge") / np.sqrt(out["ou_prediction_variance"])).clip(-2, 2) * 0.18
    out["recommended_timing"] = np.where(out["side"].eq("OVER"), "BET_EARLY", "BET_LATE")
    out["weather_cluster"] = out.apply(_weather_cluster, axis=1)
    out = out.sort_values(["game_date", "game_pk", "side"], kind="mergesort").reset_index(drop=True)
    out["ou_model_wrong"] = np.where(out["game_result"].eq("WIN"), out["ou_result_probability_formula"].lt(0.5), out["ou_result_probability_formula"].ge(0.5)).astype(int)
    out["ou_market_loss"] = out["clv_result"].eq("LOSS").astype(int)
    out["ou_model_wrong_rate_l5"] = _rolling_shifted_rate(out, ["side", "total_bucket_group"], "ou_model_wrong")
    out["ou_market_loss_rate_l5"] = _rolling_shifted_rate(out, ["side", "total_bucket_group"], "ou_market_loss")
    for col in OU_EXTRA_FEATURES_V28:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def run_ou_v28(predictions: pd.DataFrame, features: pd.DataFrame | None, cfg: MarketIntelligenceV28Config) -> dict[str, pd.DataFrame]:
    class _OUCfg:
        min_open_edge = cfg.ou_min_open_edge
        max_open_edge = cfg.ou_max_open_edge

    frame = enrich_ou_frame_v28(build_line_movement_frame(predictions, features, _OUCfg()))
    numeric_features = list(dict.fromkeys(BASE_NUMERIC_FEATURES + OU_INTERACTION_FEATURES_V26 + OU_EXTRA_FEATURES_V28 + [col for col in INJURY_NUMERIC_FEATURES if col in frame.columns]))
    train = frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)].copy()
    models = {"clv": {}, "game": {}}
    metadata = []
    for side, group in train.groupby("side", sort=True):
        for target, label in [("target_clv_win", "clv"), ("target_game_win", "game")]:
            if group[target].nunique() < 2:
                continue
            model = build_pipeline(numeric_features, OU_CATEGORICAL_FEATURES_V28)
            model.fit(group[numeric_features + OU_CATEGORICAL_FEATURES_V28], group[target].astype(int))
            models[label][side] = model
            metadata.append({"market": "OU", "model_group": side, "model": label, "target": target, "training_rows": int(len(group)), "sklearn_version": sklearn.__version__})
    scored = frame.copy()
    scored["ou_clv_probability"] = np.nan
    scored["ou_game_probability"] = np.nan
    for side, model in models["clv"].items():
        mask = scored["side"].eq(side)
        if mask.any():
            scored.loc[mask, "ou_clv_probability"] = model.predict_proba(scored.loc[mask, numeric_features + OU_CATEGORICAL_FEATURES_V28])[:, 1]
    for side, model in models["game"].items():
        mask = scored["side"].eq(side)
        if mask.any():
            scored.loc[mask, "ou_game_probability"] = model.predict_proba(scored.loc[mask, numeric_features + OU_CATEGORICAL_FEATURES_V28])[:, 1]
    scored["ou_model_disagreement"] = (scored["ou_game_probability"] - scored["ou_clv_probability"]).abs().fillna(0.0)
    scored["ou_uncertainty"] = (scored["ou_prediction_variance"] / 30.0 + scored["ou_model_disagreement"]).clip(0.001, 1.0)
    scored["ou_clv_ev"] = scored["ou_clv_probability"] * scored["payout_per_unit"] - (1.0 - scored["ou_clv_probability"])
    scored["ou_game_ev"] = scored["ou_game_probability"] * scored["payout_per_unit"] - (1.0 - scored["ou_game_probability"])
    scored["ou_blended_ev"] = 0.60 * scored["ou_game_ev"] + 0.40 * scored["ou_clv_ev"]
    scored["ou_risk_adjusted_ev"] = scored["ou_blended_ev"] * (1.0 - cfg.ou_uncertainty_penalty * scored["ou_uncertainty"]).clip(0.05, 1.0)
    scored["rank_score"] = (scored["ou_risk_adjusted_ev"] * (scored["ou_game_probability"] - 0.5).clip(lower=0.001) / np.sqrt(scored["ou_uncertainty"])).round(6)
    scored["profit_units"] = 0.0
    scored.loc[scored["game_result"].eq("WIN"), "profit_units"] = scored.loc[scored["game_result"].eq("WIN"), "payout_per_unit"]
    scored.loc[scored["game_result"].eq("LOSS"), "profit_units"] = -1.0

    gate_rows = []
    train_scored = scored[(scored["season"] >= cfg.train_start_season) & (scored["season"] <= cfg.train_end_season)]
    for (side, bucket), group in train_scored.groupby(["side", "total_bucket_group"], sort=True):
        group = group[group["ou_clv_probability"].ge(cfg.ou_min_clv_probability) & group["ou_game_probability"].ge(cfg.ou_min_game_probability) & group["ou_clv_ev"].ge(0) & group["ou_game_ev"].ge(cfg.ou_min_game_ev)]
        row = _summary_row(group, f"{side}_{bucket}", result_col="game_result")
        shrunk_clv, shrunk_roi = _shrunk_values(group, cfg.ou_segment_prior_bets, cfg.ou_segment_prior_clv)
        row.update({"side": side, "total_bucket_group": bucket, "shrunk_clv_win_rate": shrunk_clv, "shrunk_roi": shrunk_roi, "stake_multiplier": _segment_stake_multiplier(shrunk_clv, shrunk_roi, cfg)})
        row["allowed_segment"] = row["bets"] >= cfg.ou_min_segment_bets and shrunk_clv >= cfg.ou_segment_min_shrunk_clv and shrunk_roi >= cfg.ou_segment_min_shrunk_roi
        gate_rows.append(row)
    gate = pd.DataFrame(gate_rows)
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    allowed = gate[gate["allowed_segment"].astype(bool)] if not gate.empty else pd.DataFrame()
    keys = set(zip(allowed["side"].astype(str), allowed["total_bucket_group"].astype(str))) if not allowed.empty else set()
    allowed_mask = pd.Series(list(zip(test["side"].astype(str), test["total_bucket_group"].astype(str))), index=test.index).isin(keys)
    pool = test[test["ou_clv_probability"].ge(cfg.ou_min_clv_probability) & test["ou_game_probability"].ge(cfg.ou_min_game_probability) & test["ou_clv_ev"].ge(0) & test["ou_game_ev"].ge(cfg.ou_min_game_ev) & allowed_mask].copy()
    selected = []
    for date, group in pool.groupby("game_date", sort=True):
        used_weather: dict[str, int] = {}
        picks = []
        ordered = group.sort_values(["rank_score", "ou_risk_adjusted_ev", "open_edge"], ascending=[False, False, False])
        per_side = {"OVER": 0, "UNDER": 0}
        for idx, row in ordered.iterrows():
            side = str(row["side"])
            cluster = str(row.get("weather_cluster", ""))
            if per_side.get(side, 0) >= cfg.ou_max_daily_per_side:
                continue
            if cluster and used_weather.get(cluster, 0) >= cfg.max_daily_weather_cluster_exposure:
                continue
            picks.append(idx)
            per_side[side] = per_side.get(side, 0) + 1
            if cluster:
                used_weather[cluster] = used_weather.get(cluster, 0) + 1
            if len(picks) >= cfg.ou_max_daily:
                break
        selected.append(ordered.loc[picks])
    orders = pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame(columns=test.columns)
    orders = _merge_stake_multiplier(orders, gate, ["side", "total_bucket_group"])
    orders["market"] = "OU"
    orders = add_clv_snapshot_fields(orders, market="OU", mode=cfg.snapshot_mode)
    bet_snapshots, closing_snapshots = build_snapshot_tables(orders)
    return {
        "ou_scored_candidates": scored,
        "ou_orders": orders.reset_index(drop=True),
        "ou_bet_snapshots": bet_snapshots,
        "ou_closing_snapshots": closing_snapshots,
        "ou_overall": pd.DataFrame([_summary_row(orders, "ALL", result_col="game_result")]),
        "ou_segment_gate": gate,
        "ou_model_metrics": pd.concat([_ou_model_metrics(scored, cfg, "ou_clv_probability", "target_clv_win", "clv"), _ou_model_metrics(scored, cfg, "ou_game_probability", "target_game_win", "game")], ignore_index=True),
        "ou_model_metadata": pd.DataFrame(metadata),
    }


def run_v28(*, ml_scored_candidates: pd.DataFrame, ou_predictions: pd.DataFrame | None, features: pd.DataFrame | None, cfg: MarketIntelligenceV28Config) -> dict[str, pd.DataFrame]:
    outputs = run_ml_v28(ml_scored_candidates, cfg)
    if ou_predictions is not None:
        # v2.8 keeps the ML uncertainty/risk layer, but OU intentionally uses
        # the v2.7 selector after v2.8's OU uncertainty selector underperformed
        # in the 2026 through-May-1 validation window.
        outputs.update(run_ou_v27(ou_predictions, features, cfg))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLB v2.8 uncertainty, timing, lineup proxy, and dynamic segment weighting.")
    parser.add_argument("--ml-scored-candidates", required=True, type=Path)
    parser.add_argument("--ou-predictions", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-start-season", default=MarketIntelligenceV28Config.train_start_season, type=int)
    parser.add_argument("--train-end-season", default=MarketIntelligenceV28Config.train_end_season, type=int)
    parser.add_argument("--test-season", default=MarketIntelligenceV28Config.test_season, type=int)
    parser.add_argument("--snapshot-mode", choices=["historical_backtest", "live_paper"], default=MarketIntelligenceV28Config.snapshot_mode)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MarketIntelligenceV28Config(train_start_season=args.train_start_season, train_end_season=args.train_end_season, test_season=args.test_season, snapshot_mode=args.snapshot_mode)
    ml_scored = pd.read_csv(args.ml_scored_candidates, low_memory=False)
    ou_predictions = pd.read_csv(args.ou_predictions, low_memory=False) if args.ou_predictions else None
    features = pd.read_csv(args.features, low_memory=False) if args.features else None
    outputs = run_v28(ml_scored_candidates=ml_scored, ou_predictions=ou_predictions, features=features, cfg=cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {"research_version": RESEARCH_VERSION, "generated_at_utc": utc_now_iso(), "ml_scored_hash": canonical_hash(ml_scored), "ou_predictions_hash": canonical_hash(ou_predictions) if ou_predictions is not None else None, "features_hash": canonical_hash(features) if features is not None else None, "config": asdict(cfg), "outputs": {name: canonical_hash(frame) for name, frame in outputs.items()}},
    )
    print(outputs["ml_overall"].to_string(index=False))
    if "ou_overall" in outputs:
        print(outputs["ou_overall"].to_string(index=False))


if __name__ == "__main__":
    main()
