from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .governance import canonical_hash, utc_now_iso, write_manifest
from .market_intelligence_v2_5 import (
    INJURY_NUMERIC_FEATURES,
    MarketIntelligenceV25Config,
    _allowed_segment_mask,
    _ensure_injury_columns,
    _num,
    _price_band,
    _safe_divide,
    _side_injury_features,
    _summary_row,
    build_pipeline,
    enrich_ou_frame,
)
from .market_snapshots import add_clv_snapshot_fields, build_snapshot_tables
from .moneyline_classifier_v2_2 import american_implied_probability, american_profit_per_unit
from .ou_line_movement_model import BASE_CATEGORICAL_FEATURES, BASE_NUMERIC_FEATURES, build_line_movement_frame


RESEARCH_VERSION = "mlb_market_intelligence_v2_6"


@dataclass(frozen=True)
class MarketIntelligenceV26Config(MarketIntelligenceV25Config):
    ml_min_game_probability: float = 0.52
    ml_min_clv_probability: float = 0.54
    ml_min_ev_open: float = 0.04
    ml_min_ev_close: float = -0.02
    ml_segment_prior_bets: int = 30
    ml_segment_prior_clv: float = 0.50
    ml_segment_min_shrunk_clv: float = 0.515
    ml_segment_min_shrunk_roi: float = -0.01
    max_daily_team_exposure: int = 1
    ou_min_result_probability: float = 0.515
    ou_segment_prior_bets: int = 30
    ou_segment_prior_clv: float = 0.50
    ou_segment_min_shrunk_clv: float = 0.515
    snapshot_mode: str = "historical_backtest"
    research_version: str = RESEARCH_VERSION


CAUSAL_ML_FEATURES = [
    "home_rest_days",
    "away_rest_days",
    "home_games_last_3_days",
    "away_games_last_3_days",
    "home_travel_flag",
    "away_travel_flag",
    "rest_days_diff",
    "games_last_3_days_diff",
    "travel_flag_diff",
    "home_starter_rest_days",
    "away_starter_rest_days",
    "starter_rest_days_diff",
    "home_starter_recent_form_index",
    "away_starter_recent_form_index",
    "starter_recent_form_diff_for_side",
    "home_sp_xFIP",
    "away_sp_xFIP",
    "home_sp_kbb",
    "away_sp_kbb",
    "starter_xFIP_diff_for_side",
    "starter_kbb_diff_for_side",
    "home_wRC_plus_vs_hand",
    "away_wRC_plus_vs_hand",
    "offense_wRC_plus_diff_for_side",
    "team_offense_form_diff_l7_for_side",
    "team_offense_form_diff_l14_for_side",
    "team_offense_form_diff_l30_for_side",
    "combined_barrel_rate",
    "bullpen_xFIP_diff_for_side",
    "bullpen_fatigue_diff_for_side",
    "bullpen_fatigue_index",
    "park_run_factor",
    "temperature_f",
    "wind_mph",
    "humidity_pct",
    "weather_run_index",
]


ML_NUMERIC_FEATURES_V26 = [
    "model_probability",
    "probability_edge",
    "ev_open",
    "wager_price",
    "selected_current_price",
    "market_probability_for_edge",
    "selected_open_implied_probability",
    "selected_current_implied_probability",
    "selected_open_no_vig_probability",
    "market_vig",
    "pregame_implied_probability_move",
    "pregame_steam_direction",
    "pregame_stale_price_flag",
    "price_band_ordinal",
    "model_margin_for_side",
    "model_total",
    "edge_x_pregame_steam",
    "edge_x_price_band",
    "injury_x_pregame_move",
] + CAUSAL_ML_FEATURES + INJURY_NUMERIC_FEATURES

ML_CATEGORICAL_FEATURES_V26 = ["side", "favorite_group", "price_band", "venue_id"]

OU_INTERACTION_FEATURES_V26 = [
    "ou_result_probability",
    "edge_x_weather",
    "edge_x_park",
    "edge_x_bullpen_fatigue",
    "starter_gap_x_wind",
]


def _price_band_ordinal(band: pd.Series) -> pd.Series:
    mapping = {"DOG_140_PLUS": -2, "DOG_100_140": -1, "UNKNOWN": 0, "FAV_LT_160": 1, "FAV_160_220": 2, "FAV_220_PLUS": 3}
    return band.astype(str).map(mapping).fillna(0).astype(float)


def _normal_cdf(x: pd.Series) -> pd.Series:
    values = pd.to_numeric(x, errors="coerce")
    return values.map(lambda value: 0.5 * (1.0 + erf(float(value) / sqrt(2.0))) if pd.notna(value) else np.nan)


def prepare_ml_frame_v26(scored: pd.DataFrame, cfg: MarketIntelligenceV26Config) -> pd.DataFrame:
    out = _side_injury_features(scored)
    raw = _num(out, "raw_model_probability")
    if raw.isna().all():
        raw = _num(out, "model_probability")
    out["raw_model_probability"] = raw
    out["model_probability"] = (0.5 + (raw - 0.5) * cfg.probability_shrinkage).clip(0.001, 0.999)

    out["selected_price"] = _num(out, "selected_price")
    out["selected_close_price"] = out["selected_price"]
    out["selected_open_price"] = _num(out, "selected_open_price")
    out["wager_price"] = out["selected_open_price"]
    out["selected_current_price"] = _num(out, "selected_current_price")
    out["selected_current_price"] = out["selected_current_price"].fillna(out["wager_price"])

    out["selected_open_implied_probability"] = american_implied_probability(out["selected_open_price"])
    out["selected_current_implied_probability"] = american_implied_probability(out["selected_current_price"])
    out["selected_close_implied_probability"] = american_implied_probability(out["selected_close_price"])
    opponent_open = np.where(out["side"].astype(str).eq("HOME"), _num(out, "away_moneyline_open"), _num(out, "home_moneyline_open"))
    opponent_open_imp = american_implied_probability(pd.Series(opponent_open, index=out.index))
    open_sum = out["selected_open_implied_probability"] + opponent_open_imp
    out["selected_open_no_vig_probability"] = _safe_divide(out["selected_open_implied_probability"], open_sum)
    out["market_probability_for_edge"] = out["selected_open_no_vig_probability"]
    out["favorite_group"] = np.select(
        [out["market_probability_for_edge"].ge(0.5), out["market_probability_for_edge"].lt(0.5)],
        ["FAVORITE", "UNDERDOG"],
        default="UNKNOWN",
    )
    out["price_band"] = _price_band(out["wager_price"])
    out["price_band_ordinal"] = _price_band_ordinal(out["price_band"])
    out["market_vig"] = open_sum - 1.0
    out["probability_edge"] = out["model_probability"] - out["market_probability_for_edge"]
    out["payout_per_unit"] = american_profit_per_unit(out["wager_price"])
    out["close_payout_per_unit"] = american_profit_per_unit(out["selected_close_price"])
    out["ev_open"] = out["model_probability"] * out["payout_per_unit"] - (1.0 - out["model_probability"])
    out["ev_close"] = out["model_probability"] * out["close_payout_per_unit"] - (1.0 - out["model_probability"])
    out["pregame_implied_probability_move"] = out["selected_current_implied_probability"] - out["selected_open_implied_probability"]
    out["pregame_steam_direction"] = np.select(
        [out["pregame_implied_probability_move"] > 0.01, out["pregame_implied_probability_move"] < -0.01],
        [1, -1],
        default=0,
    )
    out["pregame_stale_price_flag"] = out["pregame_implied_probability_move"].abs().lt(0.0025).astype(int)
    out["edge_x_pregame_steam"] = out["probability_edge"] * out["pregame_steam_direction"]
    out["edge_x_price_band"] = out["probability_edge"] * out["price_band_ordinal"]
    out["injury_x_pregame_move"] = _num(out, "injury_severity_diff_for_side", 0.0) * out["pregame_implied_probability_move"]
    out["target_clv_win"] = out["clv_result"].eq("WIN").astype(int)
    out["target_game_win"] = out["actual_result"].eq("WIN").astype(int)
    out["profit_units"] = 0.0
    out.loc[out["actual_result"].eq("WIN"), "profit_units"] = out.loc[out["actual_result"].eq("WIN"), "payout_per_unit"]
    out.loc[out["actual_result"].eq("LOSS"), "profit_units"] = -1.0

    for col in ML_NUMERIC_FEATURES_V26:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ML_CATEGORICAL_FEATURES_V26:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str)
    return out.sort_values(["game_date", "game_pk", "side"], kind="mergesort").reset_index(drop=True)


def fit_group_models(frame: pd.DataFrame, cfg: MarketIntelligenceV26Config, target: str) -> tuple[dict[str, object], pd.DataFrame]:
    train = frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)].copy()
    train = train[train["favorite_group"].isin(["FAVORITE", "UNDERDOG"])].copy()
    models = {}
    rows = []
    for group_name, group in train.groupby("favorite_group", sort=True):
        if group[target].nunique() < 2:
            continue
        model = build_pipeline(ML_NUMERIC_FEATURES_V26, ML_CATEGORICAL_FEATURES_V26)
        model.fit(group[ML_NUMERIC_FEATURES_V26 + ML_CATEGORICAL_FEATURES_V26], group[target].astype(int))
        models[group_name] = model
        rows.append(
            {
                "market": "ML",
                "model_group": group_name,
                "target": target,
                "training_rows": int(len(group)),
                "training_hash": canonical_hash(group),
                "sklearn_version": sklearn.__version__,
            }
        )
    if not models:
        raise ValueError(f"No ML models trained for {target}")
    return models, pd.DataFrame(rows)


def score_ml_v26(frame: pd.DataFrame, clv_models: dict[str, object], game_models: dict[str, object]) -> pd.DataFrame:
    out = frame.copy()
    out["ml_clv_probability"] = np.nan
    out["ml_game_probability"] = np.nan
    for group_name, model in clv_models.items():
        mask = out["favorite_group"].eq(group_name)
        if mask.any():
            out.loc[mask, "ml_clv_probability"] = model.predict_proba(out.loc[mask, ML_NUMERIC_FEATURES_V26 + ML_CATEGORICAL_FEATURES_V26])[:, 1]
    for group_name, model in game_models.items():
        mask = out["favorite_group"].eq(group_name)
        if mask.any():
            out.loc[mask, "ml_game_probability"] = model.predict_proba(out.loc[mask, ML_NUMERIC_FEATURES_V26 + ML_CATEGORICAL_FEATURES_V26])[:, 1]
    out["ml_clv_ev"] = out["ml_clv_probability"] * out["payout_per_unit"] - (1.0 - out["ml_clv_probability"])
    out["ml_game_ev_open"] = out["ml_game_probability"] * out["payout_per_unit"] - (1.0 - out["ml_game_probability"])
    out["ml_game_ev_close"] = out["ml_game_probability"] * out["close_payout_per_unit"] - (1.0 - out["ml_game_probability"])
    out["rank_score"] = (
        out["ml_game_ev_open"]
        * np.sqrt(out["ml_game_probability"].clip(0.001, 0.999))
        * np.sqrt(out["ml_clv_probability"].clip(0.001, 0.999))
    ).round(6)
    return out


def _shrunk_segment_values(group: pd.DataFrame, cfg: MarketIntelligenceV26Config) -> tuple[float, float]:
    decisions = group["clv_result"].isin(["WIN", "LOSS"])
    clv_wins = int(group.loc[decisions, "clv_result"].eq("WIN").sum())
    clv_decisions = int(decisions.sum())
    shrunk_clv = (clv_wins + cfg.ml_segment_prior_bets * cfg.ml_segment_prior_clv) / max(clv_decisions + cfg.ml_segment_prior_bets, 1)
    shrunk_roi = float(group["profit_units"].sum()) / max(len(group) + cfg.ml_segment_prior_bets, 1)
    return shrunk_clv, shrunk_roi


def build_ml_segment_gate_v26(scored: pd.DataFrame, cfg: MarketIntelligenceV26Config) -> pd.DataFrame:
    train = scored[(scored["season"] >= cfg.train_start_season) & (scored["season"] <= cfg.train_end_season)].copy()
    edge_threshold = np.where(train["favorite_group"].eq("FAVORITE"), cfg.ml_favorite_edge, cfg.ml_underdog_edge)
    train = train[
        train["model_probability"].ge(cfg.ml_min_win_probability)
        & train["ml_game_probability"].ge(cfg.ml_min_game_probability)
        & train["probability_edge"].ge(edge_threshold)
        & train["ev_open"].ge(cfg.ml_min_ev_open)
        & train["ml_clv_probability"].ge(cfg.ml_min_clv_probability)
        & train["ml_game_ev_open"].ge(0)
        & train["ml_clv_ev"].ge(0)
    ].copy()
    rows = []
    for (group_name, band), group in train.groupby(["favorite_group", "price_band"], sort=True):
        row = _summary_row(group, f"{group_name}_{band}")
        shrunk_clv, shrunk_roi = _shrunk_segment_values(group, cfg)
        row.update({"favorite_group": group_name, "price_band": band, "shrunk_clv_win_rate": shrunk_clv, "shrunk_roi": shrunk_roi})
        allowed = (
            row["bets"] >= cfg.segment_min_bets
            and shrunk_clv >= cfg.ml_segment_min_shrunk_clv
            and shrunk_roi >= cfg.ml_segment_min_shrunk_roi
        )
        if group_name == "FAVORITE" and band == "FAV_220_PLUS":
            allowed = False
        row["allowed_segment"] = allowed
        rows.append(row)
    return pd.DataFrame(rows)


def select_ml_orders_v26(scored: pd.DataFrame, gate: pd.DataFrame, cfg: MarketIntelligenceV26Config) -> pd.DataFrame:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    edge_threshold = np.where(test["favorite_group"].eq("FAVORITE"), cfg.ml_favorite_edge, cfg.ml_underdog_edge)
    eligible = (
        test["model_probability"].ge(cfg.ml_min_win_probability)
        & test["ml_game_probability"].ge(cfg.ml_min_game_probability)
        & test["probability_edge"].ge(edge_threshold)
        & test["ev_open"].ge(cfg.ml_min_ev_open)
        & test["ml_clv_probability"].ge(cfg.ml_min_clv_probability)
        & test["ml_game_ev_open"].ge(0)
        & test["ml_clv_ev"].ge(0)
        & _allowed_segment_mask(test, gate)
    )
    pool = test[eligible].copy()
    if pool.empty:
        pool["execution_action"] = []
        return pool
    game_best = [
        group.sort_values(["rank_score", "ml_game_ev_open", "ml_clv_ev", "game_pk", "side"], ascending=[False, False, False, True, True]).head(1)
        for _, group in pool.groupby("game_pk", sort=False)
    ]
    pool = pd.concat(game_best, ignore_index=False, sort=False)
    selected = []
    for _, group in pool.groupby("game_date", sort=True):
        used_teams: dict[str, int] = {}
        picks = []
        ordered = group.sort_values(["rank_score", "ml_game_ev_open", "ml_clv_ev", "game_pk"], ascending=[False, False, False, True])
        for idx, row in ordered.iterrows():
            teams = [str(row.get("home_team", "")), str(row.get("away_team", ""))]
            if any(used_teams.get(team, 0) >= cfg.max_daily_team_exposure for team in teams if team):
                continue
            picks.append(idx)
            for team in teams:
                if team:
                    used_teams[team] = used_teams.get(team, 0) + 1
            if len(picks) >= cfg.ml_max_daily:
                break
        selected.append(ordered.loc[picks])
    orders = pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame(columns=test.columns)
    orders = orders.sort_values(["game_date", "rank_score", "game_pk"], ascending=[True, False, True]).reset_index(drop=True)
    orders["execution_action"] = "PENDING"
    for idx, _row in orders.iterrows():
        recent = orders.iloc[:idx]
        recent = recent[recent["execution_action"].isin(["BET", "THROTTLE"])].tail(cfg.ml_rolling_window)
        if len(recent) >= cfg.ml_min_rolling_bets:
            clv_decisions = recent["clv_result"].isin(["WIN", "LOSS"])
            recent_clv = recent.loc[clv_decisions, "clv_result"].eq("WIN").mean() if clv_decisions.any() else 0.0
            orders.loc[idx, "execution_action"] = "OBSERVE_ONLY" if recent_clv < cfg.ml_min_rolling_clv else "BET"
        else:
            orders.loc[idx, "execution_action"] = "BET"
    orders.loc[orders["execution_action"].eq("OBSERVE_ONLY"), "profit_units"] = 0.0
    orders["market"] = "ML"
    return add_clv_snapshot_fields(orders, market="ML", mode=cfg.snapshot_mode)


def _model_metrics(scored: pd.DataFrame, cfg: MarketIntelligenceV26Config, prob_col: str, target: str, label: str) -> pd.DataFrame:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    rows = []
    for group_name, group in test.groupby("favorite_group", sort=True):
        y = group[target].astype(int)
        p = group[prob_col]
        valid = p.notna()
        if not valid.any():
            rows.append({"market": "ML", "model_group": group_name, "model": label, "rows": int(len(group)), "scored_rows": 0, "target": target})
            continue
        y = y.loc[valid]
        p = p.loc[valid]
        row = {
            "market": "ML",
            "model_group": group_name,
            "model": label,
            "rows": int(len(group)),
            "scored_rows": int(valid.sum()),
            "target": target,
            "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p, labels=[0, 1]),
            "accuracy_at_50": accuracy_score(y, p >= 0.5),
        }
        if y.nunique() > 1:
            row["auc"] = roc_auc_score(y, p)
        rows.append(row)
    return pd.DataFrame(rows)


def run_ml_v26(ml_scored_candidates: pd.DataFrame, cfg: MarketIntelligenceV26Config) -> dict[str, pd.DataFrame]:
    frame = prepare_ml_frame_v26(ml_scored_candidates, cfg)
    clv_models, clv_metadata = fit_group_models(frame, cfg, "target_clv_win")
    game_models, game_metadata = fit_group_models(frame, cfg, "target_game_win")
    scored = score_ml_v26(frame, clv_models, game_models)
    gate = build_ml_segment_gate_v26(scored, cfg)
    orders = select_ml_orders_v26(scored, gate, cfg)
    bet_snapshots, closing_snapshots = build_snapshot_tables(orders)
    bets = orders[orders.get("execution_action", pd.Series([], dtype=object)).eq("BET")] if "execution_action" in orders else orders
    observe = orders[orders.get("execution_action", pd.Series([], dtype=object)).eq("OBSERVE_ONLY")] if "execution_action" in orders else pd.DataFrame()
    return {
        "ml_scored_candidates": scored,
        "ml_orders": orders.reset_index(drop=True),
        "ml_bet_snapshots": bet_snapshots,
        "ml_closing_snapshots": closing_snapshots,
        "ml_overall": pd.DataFrame([_summary_row(bets, "BET")]),
        "ml_observe_only": pd.DataFrame([_summary_row(observe, "OBSERVE_ONLY")]),
        "ml_segment_gate": gate,
        "ml_model_metrics": pd.concat(
            [
                _model_metrics(scored, cfg, "ml_clv_probability", "target_clv_win", "clv"),
                _model_metrics(scored, cfg, "ml_game_probability", "target_game_win", "game"),
            ],
            ignore_index=True,
        ),
        "ml_model_metadata": pd.concat([clv_metadata, game_metadata], ignore_index=True),
    }


def enrich_ou_frame_v26(frame: pd.DataFrame) -> pd.DataFrame:
    out = enrich_ou_frame(frame)
    variance = (2.25 + _num(out, "expected_total_variance", 0.0)).clip(lower=1.0, upper=25.0)
    out["ou_result_probability"] = _normal_cdf(_num(out, "open_edge") / np.sqrt(variance))
    out["edge_x_weather"] = _num(out, "open_edge") * _num(out, "weather_run_index", 0.0)
    out["edge_x_park"] = _num(out, "open_edge") * (_num(out, "park_run_factor", 1.0) - 1.0)
    out["edge_x_bullpen_fatigue"] = _num(out, "open_edge") * _num(out, "bullpen_fatigue_index", 0.0)
    out["starter_gap_x_wind"] = _num(out, "starter_xFIP_diff", 0.0).abs() * _num(out, "wind_mph", 0.0)
    for col in OU_INTERACTION_FEATURES_V26:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def run_ou_v26(predictions: pd.DataFrame, features: pd.DataFrame | None, cfg: MarketIntelligenceV26Config) -> dict[str, pd.DataFrame]:
    class _OUCfg:
        min_open_edge = cfg.ou_min_open_edge
        max_open_edge = cfg.ou_max_open_edge

    frame = build_line_movement_frame(predictions, features, _OUCfg())
    frame = enrich_ou_frame_v26(frame)
    numeric_features = list(dict.fromkeys(BASE_NUMERIC_FEATURES + OU_INTERACTION_FEATURES_V26 + [col for col in INJURY_NUMERIC_FEATURES if col in frame.columns]))
    categorical_features = BASE_CATEGORICAL_FEATURES
    train = frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)].copy()
    models = {}
    metadata = []
    for side, group in train.groupby("side", sort=True):
        if group["target_clv_win"].nunique() < 2:
            continue
        model = build_pipeline(numeric_features, categorical_features)
        model.fit(group[numeric_features + categorical_features], group["target_clv_win"].astype(int))
        models[side] = model
        metadata.append({"market": "OU", "model_group": side, "target": "target_clv_win", "training_rows": int(len(group)), "sklearn_version": sklearn.__version__})
    scored = frame.copy()
    scored["ou_clv_probability"] = np.nan
    for side, model in models.items():
        mask = scored["side"].eq(side)
        if mask.any():
            scored.loc[mask, "ou_clv_probability"] = model.predict_proba(scored.loc[mask, numeric_features + categorical_features])[:, 1]
    scored["ou_clv_ev"] = scored["ou_clv_probability"] * scored["payout_per_unit"] - (1.0 - scored["ou_clv_probability"])
    scored["rank_score"] = (scored["ou_clv_ev"] * np.sqrt(scored["ou_clv_probability"].clip(0.001, 0.999)) * np.sqrt(scored["ou_result_probability"].clip(0.001, 0.999))).round(6)
    scored["profit_units"] = 0.0
    scored.loc[scored["game_result"].eq("WIN"), "profit_units"] = scored.loc[scored["game_result"].eq("WIN"), "payout_per_unit"]
    scored.loc[scored["game_result"].eq("LOSS"), "profit_units"] = -1.0

    gate_rows = []
    train_scored = scored[(scored["season"] >= cfg.train_start_season) & (scored["season"] <= cfg.train_end_season)]
    for (side, bucket), group in train_scored.groupby(["side", "opening_total_bucket"], sort=True):
        group = group[group["ou_clv_probability"].ge(cfg.ou_min_clv_probability) & group["ou_clv_ev"].ge(0)]
        row = _summary_row(group, f"{side}_{bucket}", result_col="game_result")
        decisions = group["clv_result"].isin(["WIN", "LOSS"])
        clv_wins = int(group.loc[decisions, "clv_result"].eq("WIN").sum())
        shrunk_clv = (clv_wins + cfg.ou_segment_prior_bets * cfg.ou_segment_prior_clv) / max(int(decisions.sum()) + cfg.ou_segment_prior_bets, 1)
        row.update({"side": side, "opening_total_bucket": bucket, "shrunk_clv_win_rate": shrunk_clv})
        row["allowed_segment"] = row["bets"] >= cfg.ou_min_segment_bets and shrunk_clv >= cfg.ou_segment_min_shrunk_clv
        gate_rows.append(row)
    gate = pd.DataFrame(gate_rows)
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    if gate.empty:
        allowed_mask = pd.Series(False, index=test.index)
    else:
        allowed = gate[gate["allowed_segment"].astype(bool)]
        keys = set(zip(allowed["side"].astype(str), allowed["opening_total_bucket"].astype(float)))
        allowed_mask = pd.Series(list(zip(test["side"].astype(str), test["opening_total_bucket"].astype(float))), index=test.index).isin(keys)
    pool = test[
        test["ou_clv_probability"].ge(cfg.ou_min_clv_probability)
        & test["ou_result_probability"].ge(cfg.ou_min_result_probability)
        & test["ou_clv_ev"].ge(0)
        & allowed_mask
    ].copy()
    selected = []
    for (_date, _side), group in pool.groupby(["game_date", "side"], sort=True):
        selected.append(group.sort_values(["rank_score", "ou_clv_ev", "open_edge"], ascending=[False, False, False]).head(cfg.ou_max_daily_per_side))
    pool = pd.concat(selected, ignore_index=False, sort=False) if selected else pd.DataFrame(columns=test.columns)
    daily = [group.sort_values(["rank_score", "ou_clv_ev", "open_edge"], ascending=[False, False, False]).head(cfg.ou_max_daily) for _, group in pool.groupby("game_date", sort=True)]
    orders = pd.concat(daily, ignore_index=True, sort=False) if daily else pd.DataFrame(columns=test.columns)
    orders["market"] = "OU"
    orders = add_clv_snapshot_fields(orders, market="OU", mode=cfg.snapshot_mode)
    bet_snapshots, closing_snapshots = build_snapshot_tables(orders)

    metrics = []
    for side, group in test.groupby("side", sort=True):
        y = group["target_clv_win"].astype(int)
        p = group["ou_clv_probability"]
        row = {"market": "OU", "model_group": side, "rows": int(len(group)), "target": "target_clv_win", "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1]), "accuracy_at_50": accuracy_score(y, p >= 0.5)}
        if y.nunique() > 1:
            row["auc"] = roc_auc_score(y, p)
        metrics.append(row)
    return {
        "ou_scored_candidates": scored,
        "ou_orders": orders.reset_index(drop=True),
        "ou_bet_snapshots": bet_snapshots,
        "ou_closing_snapshots": closing_snapshots,
        "ou_overall": pd.DataFrame([_summary_row(orders, "ALL", result_col="game_result")]),
        "ou_segment_gate": gate,
        "ou_model_metrics": pd.DataFrame(metrics),
        "ou_model_metadata": pd.DataFrame(metadata),
    }


def run_v26(
    *,
    ml_scored_candidates: pd.DataFrame,
    ou_predictions: pd.DataFrame | None,
    features: pd.DataFrame | None,
    cfg: MarketIntelligenceV26Config,
) -> dict[str, pd.DataFrame]:
    outputs = run_ml_v26(ml_scored_candidates, cfg)
    if ou_predictions is not None:
        outputs.update(run_ou_v26(ou_predictions, features, cfg))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLB v2.6 market intelligence with live/paper pre-game CLV snapshots.")
    parser.add_argument("--ml-scored-candidates", required=True, type=Path)
    parser.add_argument("--ou-predictions", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-start-season", default=MarketIntelligenceV26Config.train_start_season, type=int)
    parser.add_argument("--train-end-season", default=MarketIntelligenceV26Config.train_end_season, type=int)
    parser.add_argument("--test-season", default=MarketIntelligenceV26Config.test_season, type=int)
    parser.add_argument("--ml-min-clv-probability", default=MarketIntelligenceV26Config.ml_min_clv_probability, type=float)
    parser.add_argument("--ml-min-game-probability", default=MarketIntelligenceV26Config.ml_min_game_probability, type=float)
    parser.add_argument("--ou-min-clv-probability", default=MarketIntelligenceV26Config.ou_min_clv_probability, type=float)
    parser.add_argument("--snapshot-mode", choices=["historical_backtest", "live_paper"], default=MarketIntelligenceV26Config.snapshot_mode)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MarketIntelligenceV26Config(
        train_start_season=args.train_start_season,
        train_end_season=args.train_end_season,
        test_season=args.test_season,
        ml_min_clv_probability=args.ml_min_clv_probability,
        ml_min_game_probability=args.ml_min_game_probability,
        ou_min_clv_probability=args.ou_min_clv_probability,
        snapshot_mode=args.snapshot_mode,
    )
    ml_scored = pd.read_csv(args.ml_scored_candidates, low_memory=False)
    ou_predictions = pd.read_csv(args.ou_predictions, low_memory=False) if args.ou_predictions else None
    features = pd.read_csv(args.features, low_memory=False) if args.features else None
    outputs = run_v26(ml_scored_candidates=ml_scored, ou_predictions=ou_predictions, features=features, cfg=cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "research_version": RESEARCH_VERSION,
            "generated_at_utc": utc_now_iso(),
            "ml_scored_hash": canonical_hash(ml_scored),
            "ou_predictions_hash": canonical_hash(ou_predictions) if ou_predictions is not None else None,
            "features_hash": canonical_hash(features) if features is not None else None,
            "config": asdict(cfg),
            "outputs": {name: canonical_hash(frame) for name, frame in outputs.items()},
        },
    )
    print(outputs["ml_overall"].to_string(index=False))
    if "ou_overall" in outputs:
        print(outputs["ou_overall"].to_string(index=False))


if __name__ == "__main__":
    main()
