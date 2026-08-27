from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .governance import canonical_hash, utc_now_iso, write_manifest
from .market_intelligence_v2_5 import _allowed_segment_mask, _num, _summary_row, build_pipeline
from .market_intelligence_v2_6 import (
    INJURY_NUMERIC_FEATURES,
    ML_CATEGORICAL_FEATURES_V26,
    ML_NUMERIC_FEATURES_V26,
    MarketIntelligenceV26Config,
    OU_INTERACTION_FEATURES_V26,
    _normal_cdf,
    prepare_ml_frame_v26,
    enrich_ou_frame_v26,
)
from .market_snapshots import add_clv_snapshot_fields, build_snapshot_tables
from .ou_line_movement_model import BASE_CATEGORICAL_FEATURES, BASE_NUMERIC_FEATURES, build_line_movement_frame


RESEARCH_VERSION = "mlb_market_intelligence_v2_7"


@dataclass(frozen=True)
class MarketIntelligenceV27Config(MarketIntelligenceV26Config):
    require_oof_base_predictions_for_training: bool = True
    ml_min_game_probability: float = 0.52
    ml_min_clv_probability: float = 0.54
    ml_min_ev_open: float = 0.04
    ml_segment_min_shrunk_roi: float = 0.005
    ml_segment_min_shrunk_clv: float = 0.52
    ml_fav_160_220_edge_add: float = 0.015
    ml_fav_160_220_ev_add: float = 0.015
    ml_dog_140_plus_edge_add: float = 0.015
    ml_dog_140_plus_ev_add: float = 0.010
    ou_min_game_probability: float = 0.515
    ou_min_game_ev: float = 0.0
    ou_segment_min_shrunk_roi: float = 0.0
    ou_segment_min_shrunk_clv: float = 0.52
    research_version: str = RESEARCH_VERSION


ML_CATEGORICAL_FEATURES_V27 = list(dict.fromkeys(ML_CATEGORICAL_FEATURES_V26 + ["ml_side_segment"]))


def _base_oof_mask(frame: pd.DataFrame) -> pd.Series:
    if "model_train_end_season" not in frame.columns:
        return pd.Series(False, index=frame.index)
    season = pd.to_numeric(frame["season"], errors="coerce")
    train_end = pd.to_numeric(frame["model_train_end_season"], errors="coerce")
    return train_end.notna() & season.notna() & train_end.lt(season)


def _ml_side_segment(frame: pd.DataFrame) -> pd.Series:
    favorite = frame["favorite_group"].astype(str)
    side = frame["side"].astype(str)
    home_side = side.eq("HOME")
    return pd.Series(
        np.select(
            [
                favorite.eq("FAVORITE") & home_side,
                favorite.eq("FAVORITE") & ~home_side,
                favorite.eq("UNDERDOG") & home_side,
                favorite.eq("UNDERDOG") & ~home_side,
            ],
            ["HOME_FAVORITE", "ROAD_FAVORITE", "HOME_DOG", "ROAD_DOG"],
            default="UNKNOWN",
        ),
        index=frame.index,
    )


def prepare_ml_frame_v27(scored: pd.DataFrame, cfg: MarketIntelligenceV27Config) -> pd.DataFrame:
    out = prepare_ml_frame_v26(scored, cfg)
    out["base_prediction_oof"] = _base_oof_mask(out)
    out["base_prediction_oof_status"] = np.where(out["base_prediction_oof"], "OOF", "UNKNOWN_OR_IN_SAMPLE")
    out["ml_side_segment"] = _ml_side_segment(out).astype(str)
    for col in ML_CATEGORICAL_FEATURES_V27:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str)
    return out


def _training_frame(frame: pd.DataFrame, cfg: MarketIntelligenceV27Config) -> pd.DataFrame:
    train = frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)].copy()
    if cfg.require_oof_base_predictions_for_training:
        train = train[train["base_prediction_oof"].astype(bool)].copy()
    return train[train["favorite_group"].isin(["FAVORITE", "UNDERDOG"])].copy()


def fit_group_models_v27(frame: pd.DataFrame, cfg: MarketIntelligenceV27Config, target: str) -> tuple[dict[str, object], pd.DataFrame]:
    train = _training_frame(frame, cfg)
    models = {}
    rows = []
    for group_name, group in train.groupby("favorite_group", sort=True):
        if group[target].nunique() < 2:
            continue
        model = build_pipeline(ML_NUMERIC_FEATURES_V26, ML_CATEGORICAL_FEATURES_V27)
        model.fit(group[ML_NUMERIC_FEATURES_V26 + ML_CATEGORICAL_FEATURES_V27], group[target].astype(int))
        models[group_name] = model
        rows.append(
            {
                "market": "ML",
                "model_group": group_name,
                "target": target,
                "training_rows": int(len(group)),
                "oof_training_rows": int(group["base_prediction_oof"].sum()),
                "training_hash": canonical_hash(group),
                "sklearn_version": sklearn.__version__,
            }
        )
    if not models:
        raise ValueError(f"No ML models trained for {target}")
    return models, pd.DataFrame(rows)


def score_ml_v27(frame: pd.DataFrame, clv_models: dict[str, object], game_models: dict[str, object]) -> pd.DataFrame:
    out = frame.copy()
    out["ml_clv_probability"] = np.nan
    out["ml_game_probability"] = np.nan
    for group_name, model in clv_models.items():
        mask = out["favorite_group"].eq(group_name)
        if mask.any():
            out.loc[mask, "ml_clv_probability"] = model.predict_proba(out.loc[mask, ML_NUMERIC_FEATURES_V26 + ML_CATEGORICAL_FEATURES_V27])[:, 1]
    for group_name, model in game_models.items():
        mask = out["favorite_group"].eq(group_name)
        if mask.any():
            out.loc[mask, "ml_game_probability"] = model.predict_proba(out.loc[mask, ML_NUMERIC_FEATURES_V26 + ML_CATEGORICAL_FEATURES_V27])[:, 1]
    out["ml_clv_ev"] = out["ml_clv_probability"] * out["payout_per_unit"] - (1.0 - out["ml_clv_probability"])
    out["ml_game_ev_open"] = out["ml_game_probability"] * out["payout_per_unit"] - (1.0 - out["ml_game_probability"])
    out["ml_game_ev_close"] = out["ml_game_probability"] * out["close_payout_per_unit"] - (1.0 - out["ml_game_probability"])
    out["rank_score"] = (0.60 * out["ml_game_ev_open"] + 0.40 * out["ml_clv_ev"]).round(6)
    return out


def _ml_thresholds(frame: pd.DataFrame, cfg: MarketIntelligenceV27Config) -> tuple[pd.Series, pd.Series]:
    edge = pd.Series(np.where(frame["favorite_group"].eq("FAVORITE"), cfg.ml_favorite_edge, cfg.ml_underdog_edge), index=frame.index, dtype=float)
    ev = pd.Series(cfg.ml_min_ev_open, index=frame.index, dtype=float)
    fav_mid = frame["favorite_group"].eq("FAVORITE") & frame["price_band"].eq("FAV_160_220")
    dog_big = frame["favorite_group"].eq("UNDERDOG") & frame["price_band"].eq("DOG_140_PLUS")
    fav_extreme = frame["price_band"].eq("FAV_220_PLUS")
    edge.loc[fav_mid] += cfg.ml_fav_160_220_edge_add
    ev.loc[fav_mid] += cfg.ml_fav_160_220_ev_add
    edge.loc[dog_big] += cfg.ml_dog_140_plus_edge_add
    ev.loc[dog_big] += cfg.ml_dog_140_plus_ev_add
    edge.loc[fav_extreme] = np.inf
    ev.loc[fav_extreme] = np.inf
    return edge, ev


def _shrunk_values(group: pd.DataFrame, prior_bets: int, prior_clv: float) -> tuple[float, float]:
    decisions = group["clv_result"].isin(["WIN", "LOSS"])
    clv_wins = int(group.loc[decisions, "clv_result"].eq("WIN").sum())
    shrunk_clv = (clv_wins + prior_bets * prior_clv) / max(int(decisions.sum()) + prior_bets, 1)
    shrunk_roi = float(group["profit_units"].sum()) / max(len(group) + prior_bets, 1)
    return shrunk_clv, shrunk_roi


def build_ml_segment_gate_v27(scored: pd.DataFrame, cfg: MarketIntelligenceV27Config) -> pd.DataFrame:
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
        row.update(
            {
                "ml_side_segment": segment,
                "price_band": band,
                "favorite_group": "FAVORITE" if "FAVORITE" in segment else "UNDERDOG" if "DOG" in segment else "UNKNOWN",
                "shrunk_clv_win_rate": shrunk_clv,
                "shrunk_roi": shrunk_roi,
                "segment_mode": "ALLOW",
            }
        )
        allowed = (
            row["bets"] >= cfg.segment_min_bets
            and shrunk_clv >= cfg.ml_segment_min_shrunk_clv
            and shrunk_roi >= cfg.ml_segment_min_shrunk_roi
            and band != "FAV_220_PLUS"
        )
        if row["roi"] > 0 and row["clv_win_rate"] < 0.50:
            row["segment_mode"] = "OBSERVE_ONLY_NEGATIVE_CLV"
            allowed = False
        row["allowed_segment"] = allowed
        rows.append(row)
    return pd.DataFrame(rows)


def _allowed_ml_segment_mask(frame: pd.DataFrame, gate: pd.DataFrame) -> pd.Series:
    if gate.empty:
        return pd.Series(False, index=frame.index)
    allowed = gate[gate["allowed_segment"].astype(bool)]
    keys = set(zip(allowed["ml_side_segment"].astype(str), allowed["price_band"].astype(str)))
    return pd.Series(list(zip(frame["ml_side_segment"].astype(str), frame["price_band"].astype(str))), index=frame.index).isin(keys)


def select_ml_orders_v27(scored: pd.DataFrame, gate: pd.DataFrame, cfg: MarketIntelligenceV27Config) -> pd.DataFrame:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    edge_threshold, ev_threshold = _ml_thresholds(test, cfg)
    eligible = (
        test["model_probability"].ge(cfg.ml_min_win_probability)
        & test["ml_game_probability"].ge(cfg.ml_min_game_probability)
        & test["probability_edge"].ge(edge_threshold)
        & test["ev_open"].ge(ev_threshold)
        & test["ml_clv_probability"].ge(cfg.ml_min_clv_probability)
        & test["ml_game_ev_open"].ge(0)
        & test["ml_clv_ev"].ge(0)
        & _allowed_ml_segment_mask(test, gate)
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


def _ml_model_metrics(scored: pd.DataFrame, cfg: MarketIntelligenceV27Config, prob_col: str, target: str, label: str) -> pd.DataFrame:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    rows = []
    for segment, group in test.groupby("ml_side_segment", sort=True):
        y = group[target].astype(int)
        p = group[prob_col]
        valid = p.notna()
        if not valid.any():
            rows.append({"market": "ML", "model_group": segment, "model": label, "rows": int(len(group)), "scored_rows": 0, "target": target})
            continue
        y = y.loc[valid]
        p = p.loc[valid]
        row = {
            "market": "ML",
            "model_group": segment,
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


def run_ml_v27(ml_scored_candidates: pd.DataFrame, cfg: MarketIntelligenceV27Config) -> dict[str, pd.DataFrame]:
    frame = prepare_ml_frame_v27(ml_scored_candidates, cfg)
    clv_models, clv_metadata = fit_group_models_v27(frame, cfg, "target_clv_win")
    game_models, game_metadata = fit_group_models_v27(frame, cfg, "target_game_win")
    scored = score_ml_v27(frame, clv_models, game_models)
    gate = build_ml_segment_gate_v27(scored, cfg)
    orders = select_ml_orders_v27(scored, gate, cfg)
    bet_snapshots, closing_snapshots = build_snapshot_tables(orders)
    bets = orders[orders.get("execution_action", pd.Series([], dtype=object)).eq("BET")] if "execution_action" in orders else orders
    observe = orders[orders.get("execution_action", pd.Series([], dtype=object)).eq("OBSERVE_ONLY")] if "execution_action" in orders else pd.DataFrame()
    oof_audit = pd.DataFrame(
        [
            {
                "train_start_season": cfg.train_start_season,
                "train_end_season": cfg.train_end_season,
                "test_season": cfg.test_season,
                "training_rows_total": int(len(frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)])),
                "training_rows_oof": int(_training_frame(frame, cfg)["base_prediction_oof"].sum()),
                "test_rows_unknown_oof": int((frame["season"].eq(cfg.test_season) & ~frame["base_prediction_oof"]).sum()),
                "require_oof_base_predictions_for_training": cfg.require_oof_base_predictions_for_training,
            }
        ]
    )
    return {
        "ml_scored_candidates": scored,
        "ml_orders": orders.reset_index(drop=True),
        "ml_bet_snapshots": bet_snapshots,
        "ml_closing_snapshots": closing_snapshots,
        "ml_overall": pd.DataFrame([_summary_row(bets, "BET")]),
        "ml_observe_only": pd.DataFrame([_summary_row(observe, "OBSERVE_ONLY")]),
        "ml_segment_gate": gate,
        "ml_oof_audit": oof_audit,
        "ml_model_metrics": pd.concat(
            [
                _ml_model_metrics(scored, cfg, "ml_clv_probability", "target_clv_win", "clv"),
                _ml_model_metrics(scored, cfg, "ml_game_probability", "target_game_win", "game"),
            ],
            ignore_index=True,
        ),
        "ml_model_metadata": pd.concat([clv_metadata, game_metadata], ignore_index=True),
    }


def _total_bucket_group(opening_total: pd.Series) -> pd.Series:
    total = pd.to_numeric(opening_total, errors="coerce")
    return pd.Series(np.select([total <= 7.5, total >= 9.5], ["LOW", "HIGH"], default="NORMAL"), index=opening_total.index)


def _ou_model_metrics(scored: pd.DataFrame, cfg: MarketIntelligenceV27Config, prob_col: str, target: str, label: str) -> pd.DataFrame:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    rows = []
    for side, group in test.groupby("side", sort=True):
        y = group[target].astype(int)
        p = group[prob_col]
        row = {"market": "OU", "model_group": side, "model": label, "rows": int(len(group)), "target": target}
        if p.notna().any():
            row.update({"brier": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1]), "accuracy_at_50": accuracy_score(y, p >= 0.5)})
            if y.nunique() > 1:
                row["auc"] = roc_auc_score(y, p)
        rows.append(row)
    return pd.DataFrame(rows)


def run_ou_v27(predictions: pd.DataFrame, features: pd.DataFrame | None, cfg: MarketIntelligenceV27Config) -> dict[str, pd.DataFrame]:
    class _OUCfg:
        min_open_edge = cfg.ou_min_open_edge
        max_open_edge = cfg.ou_max_open_edge

    frame = build_line_movement_frame(predictions, features, _OUCfg())
    frame = enrich_ou_frame_v26(frame)
    frame["target_game_win"] = frame["game_result"].eq("WIN").astype(int)
    frame["total_bucket_group"] = _total_bucket_group(frame["opening_total"]).astype(str)
    numeric_features = list(dict.fromkeys(BASE_NUMERIC_FEATURES + OU_INTERACTION_FEATURES_V26 + [col for col in INJURY_NUMERIC_FEATURES if col in frame.columns]))
    categorical_features = list(dict.fromkeys(BASE_CATEGORICAL_FEATURES + ["total_bucket_group"]))
    train = frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)].copy()
    models: dict[str, dict[str, object]] = {"clv": {}, "game": {}}
    metadata = []
    for side, group in train.groupby("side", sort=True):
        for target, label in [("target_clv_win", "clv"), ("target_game_win", "game")]:
            if group[target].nunique() < 2:
                continue
            model = build_pipeline(numeric_features, categorical_features)
            model.fit(group[numeric_features + categorical_features], group[target].astype(int))
            models[label][side] = model
            metadata.append({"market": "OU", "model_group": side, "model": label, "target": target, "training_rows": int(len(group)), "sklearn_version": sklearn.__version__})
    scored = frame.copy()
    scored["ou_clv_probability"] = np.nan
    scored["ou_game_probability"] = np.nan
    for side, model in models["clv"].items():
        mask = scored["side"].eq(side)
        if mask.any():
            scored.loc[mask, "ou_clv_probability"] = model.predict_proba(scored.loc[mask, numeric_features + categorical_features])[:, 1]
    for side, model in models["game"].items():
        mask = scored["side"].eq(side)
        if mask.any():
            scored.loc[mask, "ou_game_probability"] = model.predict_proba(scored.loc[mask, numeric_features + categorical_features])[:, 1]
    scored["ou_clv_ev"] = scored["ou_clv_probability"] * scored["payout_per_unit"] - (1.0 - scored["ou_clv_probability"])
    scored["ou_game_ev"] = scored["ou_game_probability"] * scored["payout_per_unit"] - (1.0 - scored["ou_game_probability"])
    scored["rank_score"] = (0.60 * scored["ou_game_ev"] + 0.40 * scored["ou_clv_ev"]).round(6)
    scored["profit_units"] = 0.0
    scored.loc[scored["game_result"].eq("WIN"), "profit_units"] = scored.loc[scored["game_result"].eq("WIN"), "payout_per_unit"]
    scored.loc[scored["game_result"].eq("LOSS"), "profit_units"] = -1.0

    gate_rows = []
    train_scored = scored[(scored["season"] >= cfg.train_start_season) & (scored["season"] <= cfg.train_end_season)]
    for (side, bucket), group in train_scored.groupby(["side", "total_bucket_group"], sort=True):
        group = group[
            group["ou_clv_probability"].ge(cfg.ou_min_clv_probability)
            & group["ou_game_probability"].ge(cfg.ou_min_game_probability)
            & group["ou_clv_ev"].ge(0)
            & group["ou_game_ev"].ge(cfg.ou_min_game_ev)
        ]
        row = _summary_row(group, f"{side}_{bucket}", result_col="game_result")
        shrunk_clv, shrunk_roi = _shrunk_values(group, cfg.ou_segment_prior_bets, cfg.ou_segment_prior_clv)
        row.update({"side": side, "total_bucket_group": bucket, "shrunk_clv_win_rate": shrunk_clv, "shrunk_roi": shrunk_roi})
        row["allowed_segment"] = row["bets"] >= cfg.ou_min_segment_bets and shrunk_clv >= cfg.ou_segment_min_shrunk_clv and shrunk_roi >= cfg.ou_segment_min_shrunk_roi
        gate_rows.append(row)
    gate = pd.DataFrame(gate_rows)
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    if gate.empty:
        allowed_mask = pd.Series(False, index=test.index)
    else:
        allowed = gate[gate["allowed_segment"].astype(bool)]
        keys = set(zip(allowed["side"].astype(str), allowed["total_bucket_group"].astype(str)))
        allowed_mask = pd.Series(list(zip(test["side"].astype(str), test["total_bucket_group"].astype(str))), index=test.index).isin(keys)
    pool = test[
        test["ou_clv_probability"].ge(cfg.ou_min_clv_probability)
        & test["ou_game_probability"].ge(cfg.ou_min_game_probability)
        & test["ou_clv_ev"].ge(0)
        & test["ou_game_ev"].ge(cfg.ou_min_game_ev)
        & allowed_mask
    ].copy()
    selected = []
    for (_date, _side), group in pool.groupby(["game_date", "side"], sort=True):
        selected.append(group.sort_values(["rank_score", "ou_game_ev", "ou_clv_ev", "open_edge"], ascending=[False, False, False, False]).head(cfg.ou_max_daily_per_side))
    pool = pd.concat(selected, ignore_index=False, sort=False) if selected else pd.DataFrame(columns=test.columns)
    daily = [group.sort_values(["rank_score", "ou_game_ev", "ou_clv_ev", "open_edge"], ascending=[False, False, False, False]).head(cfg.ou_max_daily) for _, group in pool.groupby("game_date", sort=True)]
    orders = pd.concat(daily, ignore_index=True, sort=False) if daily else pd.DataFrame(columns=test.columns)
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
        "ou_model_metrics": pd.concat(
            [
                _ou_model_metrics(scored, cfg, "ou_clv_probability", "target_clv_win", "clv"),
                _ou_model_metrics(scored, cfg, "ou_game_probability", "target_game_win", "game"),
            ],
            ignore_index=True,
        ),
        "ou_model_metadata": pd.DataFrame(metadata),
    }


def run_v27(
    *,
    ml_scored_candidates: pd.DataFrame,
    ou_predictions: pd.DataFrame | None,
    features: pd.DataFrame | None,
    cfg: MarketIntelligenceV27Config,
) -> dict[str, pd.DataFrame]:
    outputs = run_ml_v27(ml_scored_candidates, cfg)
    if ou_predictions is not None:
        outputs.update(run_ou_v27(ou_predictions, features, cfg))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLB v2.7 OOF-guarded ML + OU game-result market intelligence.")
    parser.add_argument("--ml-scored-candidates", required=True, type=Path)
    parser.add_argument("--ou-predictions", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-start-season", default=MarketIntelligenceV27Config.train_start_season, type=int)
    parser.add_argument("--train-end-season", default=MarketIntelligenceV27Config.train_end_season, type=int)
    parser.add_argument("--test-season", default=MarketIntelligenceV27Config.test_season, type=int)
    parser.add_argument("--ml-min-clv-probability", default=MarketIntelligenceV27Config.ml_min_clv_probability, type=float)
    parser.add_argument("--ml-min-game-probability", default=MarketIntelligenceV27Config.ml_min_game_probability, type=float)
    parser.add_argument("--ou-min-clv-probability", default=MarketIntelligenceV27Config.ou_min_clv_probability, type=float)
    parser.add_argument("--ou-min-game-probability", default=MarketIntelligenceV27Config.ou_min_game_probability, type=float)
    parser.add_argument("--snapshot-mode", choices=["historical_backtest", "live_paper"], default=MarketIntelligenceV27Config.snapshot_mode)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MarketIntelligenceV27Config(
        train_start_season=args.train_start_season,
        train_end_season=args.train_end_season,
        test_season=args.test_season,
        ml_min_clv_probability=args.ml_min_clv_probability,
        ml_min_game_probability=args.ml_min_game_probability,
        ou_min_clv_probability=args.ou_min_clv_probability,
        ou_min_game_probability=args.ou_min_game_probability,
        snapshot_mode=args.snapshot_mode,
    )
    ml_scored = pd.read_csv(args.ml_scored_candidates, low_memory=False)
    ou_predictions = pd.read_csv(args.ou_predictions, low_memory=False) if args.ou_predictions else None
    features = pd.read_csv(args.features, low_memory=False) if args.features else None
    outputs = run_v27(ml_scored_candidates=ml_scored, ou_predictions=ou_predictions, features=features, cfg=cfg)
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
