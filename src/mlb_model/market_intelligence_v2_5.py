from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

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
from .moneyline_classifier_v2_2 import american_implied_probability, american_profit_per_unit
from .ou_line_movement_model import build_line_movement_frame


RESEARCH_VERSION = "mlb_market_intelligence_v2_5"


@dataclass(frozen=True)
class MarketIntelligenceV25Config:
    train_start_season: int = 2021
    train_end_season: int = 2025
    test_season: int = 2026
    probability_shrinkage: float = 0.80
    ml_min_win_probability: float = 0.52
    ml_min_clv_probability: float = 0.53
    ml_favorite_edge: float = 0.04
    ml_underdog_edge: float = 0.06
    ml_min_ev: float = 0.04
    ml_max_daily: int = 4
    ml_rolling_window: int = 25
    ml_min_rolling_clv: float = 0.50
    ml_min_rolling_bets: int = 12
    segment_min_bets: int = 20
    segment_min_roi: float = 0.0
    segment_min_clv: float = 0.50
    ou_min_clv_probability: float = 0.53
    ou_min_open_edge: float = 0.75
    ou_max_open_edge: float = 8.0
    ou_max_daily: int = 4
    ou_max_daily_per_side: int = 3
    ou_min_segment_bets: int = 20
    ou_min_segment_clv: float = 0.50
    research_version: str = RESEARCH_VERSION


INJURY_NUMERIC_FEATURES = [
    "home_injury_count",
    "away_injury_count",
    "home_sp_injury_count",
    "away_sp_injury_count",
    "home_rp_injury_count",
    "away_rp_injury_count",
    "home_bat_injury_count",
    "away_bat_injury_count",
    "home_injury_severity_sum",
    "away_injury_severity_sum",
    "injury_count_diff",
    "sp_injury_count_diff",
    "rp_injury_count_diff",
    "bat_injury_count_diff",
    "injury_severity_diff",
    "injury_count_diff_for_side",
    "sp_injury_count_diff_for_side",
    "rp_injury_count_diff_for_side",
    "bat_injury_count_diff_for_side",
    "injury_severity_diff_for_side",
]


ML_NUMERIC_FEATURES = [
    "model_probability",
    "probability_edge",
    "ev",
    "wager_price",
    "market_probability_for_edge",
    "selected_open_price",
    "selected_open_implied_probability",
    "selected_open_no_vig_probability",
    "market_vig",
    "model_margin_for_side",
    "model_total",
] + INJURY_NUMERIC_FEATURES

ML_CATEGORICAL_FEATURES = ["side", "favorite_group", "price_band", "venue_id"]


OU_EXTRA_NUMERIC_FEATURES = [
    "expected_total_variance",
    "home_bullpen_ip_proxy_l1d",
    "away_bullpen_ip_proxy_l1d",
    "home_bullpen_ip_proxy_l5d",
    "away_bullpen_ip_proxy_l5d",
    "home_bullpen_fatigue_rate_l1d",
    "away_bullpen_fatigue_rate_l1d",
    "home_bullpen_fatigue_rate_l5d",
    "away_bullpen_fatigue_rate_l5d",
    "team_plate_appearances_proxy",
    "umpire_run_factor",
    "umpire_strike_zone_index",
] + INJURY_NUMERIC_FEATURES


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _price_band(price: pd.Series) -> pd.Series:
    p = pd.to_numeric(price, errors="coerce")
    out = pd.Series("UNKNOWN", index=price.index, dtype=object)
    out.loc[p.between(-159.999, -100.0, inclusive="both")] = "FAV_LT_160"
    out.loc[p.between(-219.999, -160.0, inclusive="both")] = "FAV_160_220"
    out.loc[p <= -220.0] = "FAV_220_PLUS"
    out.loc[p.between(100.0, 139.999, inclusive="both")] = "DOG_100_140"
    out.loc[p >= 140.0] = "DOG_140_PLUS"
    return out


def _ensure_injury_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "home_injury_count": "home_injury_injury_count",
        "away_injury_count": "away_injury_injury_count",
        "home_sp_injury_count": "home_injury_sp_injury_count",
        "away_sp_injury_count": "away_injury_sp_injury_count",
        "home_rp_injury_count": "home_injury_rp_injury_count",
        "away_rp_injury_count": "away_injury_rp_injury_count",
        "home_bat_injury_count": "home_injury_bat_injury_count",
        "away_bat_injury_count": "away_injury_bat_injury_count",
        "home_injury_severity_sum": "home_injury_injury_severity_sum",
        "away_injury_severity_sum": "away_injury_injury_severity_sum",
    }
    for target, source in aliases.items():
        if target not in out.columns and source in out.columns:
            out[target] = out[source]
    for col in INJURY_NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "injury_count_diff" not in df.columns:
        out["injury_count_diff"] = out["home_injury_count"] - out["away_injury_count"]
    if "sp_injury_count_diff" not in df.columns:
        out["sp_injury_count_diff"] = out["home_sp_injury_count"] - out["away_sp_injury_count"]
    if "rp_injury_count_diff" not in df.columns:
        out["rp_injury_count_diff"] = out["home_rp_injury_count"] - out["away_rp_injury_count"]
    if "bat_injury_count_diff" not in df.columns:
        out["bat_injury_count_diff"] = out["home_bat_injury_count"] - out["away_bat_injury_count"]
    if "injury_severity_diff" not in df.columns:
        out["injury_severity_diff"] = out["home_injury_severity_sum"] - out["away_injury_severity_sum"]
    return out


def _side_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_injury_columns(df)
    sign = np.where(out["side"].astype(str).eq("HOME"), 1.0, -1.0)
    for base in ["injury_count", "sp_injury_count", "rp_injury_count", "bat_injury_count", "injury_severity"]:
        diff_col = f"{base}_diff" if base != "injury_severity" else "injury_severity_diff"
        if diff_col in out.columns:
            out[f"{base}_diff_for_side"] = sign * pd.to_numeric(out[diff_col], errors="coerce").fillna(0.0)
    for col in INJURY_NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
    return out


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())])
    categorical = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([("numeric", numeric, numeric_features), ("categorical", categorical, categorical_features)])
    return Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def prepare_ml_frame(scored: pd.DataFrame, cfg: MarketIntelligenceV25Config) -> pd.DataFrame:
    out = _side_injury_features(scored)
    raw = _num(out, "raw_model_probability")
    if raw.isna().all():
        raw = _num(out, "model_probability")
    out["raw_model_probability"] = raw
    out["model_probability"] = (0.5 + (raw - 0.5) * cfg.probability_shrinkage).clip(0.001, 0.999)
    out["selected_price"] = _num(out, "selected_price")
    out["selected_open_price"] = _num(out, "selected_open_price")
    out["wager_price"] = out["selected_open_price"].fillna(out["selected_price"])
    out["selected_implied_probability"] = american_implied_probability(out["selected_price"])
    out["selected_open_implied_probability"] = american_implied_probability(out["selected_open_price"])
    opponent_open = np.where(
        out["side"].astype(str).eq("HOME"),
        _num(out, "away_moneyline_open"),
        _num(out, "home_moneyline_open"),
    )
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
    out["market_vig"] = (open_sum - 1.0).fillna(_num(out, "market_vig"))
    out["probability_edge"] = out["model_probability"] - out["market_probability_for_edge"]
    out["payout_per_unit"] = american_profit_per_unit(out["wager_price"])
    out["ev"] = out["model_probability"] * out["payout_per_unit"] - (1.0 - out["model_probability"])
    out["implied_probability_move"] = out["selected_implied_probability"] - out["selected_open_implied_probability"]
    out["abs_implied_probability_move"] = out["implied_probability_move"].abs()
    out["market_moved_toward_side"] = out["implied_probability_move"].gt(0.0025).astype(int)
    out["steam_direction"] = np.select(
        [out["implied_probability_move"] > 0.01, out["implied_probability_move"] < -0.01],
        [1, -1],
        default=0,
    )
    out["stale_price_flag"] = out["abs_implied_probability_move"].lt(0.0025).astype(int)
    out["favorite_flip"] = (
        _num(out, "selected_no_vig_probability").ge(0.5) != out["selected_open_no_vig_probability"].ge(0.5)
    ).astype(int)
    out["target_clv_win"] = out["clv_result"].eq("WIN").astype(int)
    out["target_game_win"] = out["actual_result"].eq("WIN").astype(int)
    for col in ML_NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ML_CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str)
    return out.sort_values(["game_date", "game_pk", "side"], kind="mergesort").reset_index(drop=True)


def fit_ml_clv_models(frame: pd.DataFrame, cfg: MarketIntelligenceV25Config) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    train = frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)].copy()
    models = {}
    metadata = []
    for group_name, group in train.groupby("favorite_group", sort=True):
        if group["target_clv_win"].nunique() < 2:
            continue
        model = build_pipeline(ML_NUMERIC_FEATURES, ML_CATEGORICAL_FEATURES)
        model.fit(group[ML_NUMERIC_FEATURES + ML_CATEGORICAL_FEATURES], group["target_clv_win"])
        models[group_name] = model
        metadata.append(
            {
                "market": "ML",
                "model_group": group_name,
                "target": "target_clv_win",
                "training_rows": int(len(group)),
                "training_hash": canonical_hash(group),
                "sklearn_version": sklearn.__version__,
            }
        )
    if not models:
        raise ValueError("No ML CLV models were trained")
    return models, pd.DataFrame(metadata)


def score_ml_clv(frame: pd.DataFrame, models: dict[str, Pipeline]) -> pd.DataFrame:
    out = frame.copy()
    out["ml_clv_probability"] = np.nan
    for group_name, model in models.items():
        mask = out["favorite_group"].eq(group_name)
        if mask.any():
            out.loc[mask, "ml_clv_probability"] = model.predict_proba(out.loc[mask, ML_NUMERIC_FEATURES + ML_CATEGORICAL_FEATURES])[:, 1]
    out["ml_clv_ev"] = out["ml_clv_probability"] * out["payout_per_unit"] - (1.0 - out["ml_clv_probability"])
    out["rank_score"] = (0.65 * out["ml_clv_ev"] + 0.35 * out["ev"]).round(6)
    out["profit_units"] = 0.0
    out.loc[out["actual_result"].eq("WIN"), "profit_units"] = out.loc[out["actual_result"].eq("WIN"), "payout_per_unit"]
    out.loc[out["actual_result"].eq("LOSS"), "profit_units"] = -1.0
    return out


def _summary_row(df: pd.DataFrame, label: str, result_col: str = "actual_result") -> dict:
    wins = int(df[result_col].eq("WIN").sum()) if result_col in df else 0
    losses = int(df[result_col].eq("LOSS").sum()) if result_col in df else 0
    pushes = int(df[result_col].eq("PUSH").sum()) if result_col in df else 0
    decisions = wins + losses
    profit = float(df["profit_units"].sum()) if "profit_units" in df and len(df) else 0.0
    clv_decisions = df["clv_result"].isin(["WIN", "LOSS"]) if "clv_result" in df and len(df) else pd.Series(dtype=bool)
    return {
        "segment": label,
        "bets": int(len(df)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": wins / decisions if decisions else 0.0,
        "profit_units": profit,
        "roi": profit / len(df) if len(df) else 0.0,
        "clv_wins": int(df["clv_result"].eq("WIN").sum()) if "clv_result" in df and len(df) else 0,
        "clv_losses": int(df["clv_result"].eq("LOSS").sum()) if "clv_result" in df and len(df) else 0,
        "clv_win_rate": float(df.loc[clv_decisions, "clv_result"].eq("WIN").mean()) if len(clv_decisions) and clv_decisions.any() else 0.0,
    }


def build_ml_segment_gate(scored: pd.DataFrame, cfg: MarketIntelligenceV25Config) -> pd.DataFrame:
    train = scored[(scored["season"] >= cfg.train_start_season) & (scored["season"] <= cfg.train_end_season)].copy()
    edge_threshold = np.where(train["favorite_group"].eq("FAVORITE"), cfg.ml_favorite_edge, cfg.ml_underdog_edge)
    train = train[
        train["probability_edge"].ge(edge_threshold)
        & train["ev"].ge(cfg.ml_min_ev)
        & train["ml_clv_probability"].ge(cfg.ml_min_clv_probability)
        & train["ml_clv_ev"].ge(0)
    ].copy()
    rows = []
    for (group_name, band), group in train.groupby(["favorite_group", "price_band"], sort=True):
        row = _summary_row(group, f"{group_name}_{band}")
        row.update({"favorite_group": group_name, "price_band": band})
        allowed = row["bets"] >= cfg.segment_min_bets and row["roi"] >= cfg.segment_min_roi and row["clv_win_rate"] >= cfg.segment_min_clv
        if group_name == "FAVORITE" and band != "FAV_LT_160":
            allowed = allowed and row["roi"] > 0 and row["clv_win_rate"] >= cfg.segment_min_clv
        if group_name == "UNDERDOG":
            allowed = allowed and row["clv_win_rate"] >= cfg.segment_min_clv
        row["allowed_segment"] = allowed
        rows.append(row)
    return pd.DataFrame(rows)


def _allowed_segment_mask(frame: pd.DataFrame, gate: pd.DataFrame) -> pd.Series:
    if gate.empty:
        return pd.Series(False, index=frame.index)
    allowed = gate[gate["allowed_segment"].astype(bool)]
    keys = set(zip(allowed["favorite_group"].astype(str), allowed["price_band"].astype(str)))
    return pd.Series(list(zip(frame["favorite_group"].astype(str), frame["price_band"].astype(str))), index=frame.index).isin(keys)


def select_ml_orders(scored: pd.DataFrame, gate: pd.DataFrame, cfg: MarketIntelligenceV25Config) -> pd.DataFrame:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    edge_threshold = np.where(test["favorite_group"].eq("FAVORITE"), cfg.ml_favorite_edge, cfg.ml_underdog_edge)
    eligible = (
        test["model_probability"].ge(cfg.ml_min_win_probability)
        & test["probability_edge"].ge(edge_threshold)
        & test["ev"].ge(cfg.ml_min_ev)
        & test["ml_clv_probability"].ge(cfg.ml_min_clv_probability)
        & test["ml_clv_ev"].ge(0)
        & _allowed_segment_mask(test, gate)
    )
    pool = test[eligible].copy()
    game_best = []
    for _, group in pool.groupby("game_pk", sort=False):
        game_best.append(group.sort_values(["rank_score", "ml_clv_ev", "ev", "game_pk", "side"], ascending=[False, False, False, True, True]).head(1))
    pool = pd.concat(game_best, ignore_index=False, sort=False) if game_best else pd.DataFrame(columns=test.columns)
    selected = []
    for _, group in pool.groupby("game_date", sort=True):
        selected.append(group.sort_values(["rank_score", "ml_clv_ev", "ev", "game_pk"], ascending=[False, False, False, True]).head(cfg.ml_max_daily))
    orders = pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame(columns=test.columns)
    if orders.empty:
        orders["execution_action"] = []
        return orders

    orders = orders.sort_values(["game_date", "rank_score", "game_pk"], ascending=[True, False, True]).reset_index(drop=True)
    orders["execution_action"] = "PENDING"
    for idx, row in orders.iterrows():
        previous = orders.iloc[:idx]
        recent = previous[previous["execution_action"].isin(["BET", "THROTTLE"])].tail(cfg.ml_rolling_window)
        if len(recent) >= cfg.ml_min_rolling_bets:
            clv_decisions = recent["clv_result"].isin(["WIN", "LOSS"])
            recent_clv = recent.loc[clv_decisions, "clv_result"].eq("WIN").mean() if clv_decisions.any() else 0.0
            orders.loc[idx, "execution_action"] = "OBSERVE_ONLY" if recent_clv < cfg.ml_min_rolling_clv else "BET"
        else:
            orders.loc[idx, "execution_action"] = "BET"
    orders.loc[orders["execution_action"].eq("OBSERVE_ONLY"), "profit_units"] = 0.0
    return orders


def ml_metrics(scored: pd.DataFrame, cfg: MarketIntelligenceV25Config) -> pd.DataFrame:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    rows = []
    for group_name, group in test.groupby("favorite_group", sort=True):
        y = group["target_clv_win"].astype(int)
        p = group["ml_clv_probability"]
        valid = p.notna()
        if not valid.any():
            rows.append(
                {
                    "market": "ML",
                    "model_group": group_name,
                    "rows": int(len(group)),
                    "scored_rows": 0,
                    "target": "target_clv_win",
                    "brier": np.nan,
                    "log_loss": np.nan,
                    "accuracy_at_50": np.nan,
                }
            )
            continue
        y = y.loc[valid]
        p = p.loc[valid]
        row = {
            "market": "ML",
            "model_group": group_name,
            "rows": int(len(group)),
            "scored_rows": int(valid.sum()),
            "target": "target_clv_win",
            "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p, labels=[0, 1]),
            "accuracy_at_50": accuracy_score(y, p >= 0.5),
        }
        if y.nunique() > 1:
            row["auc"] = roc_auc_score(y, p)
        rows.append(row)
    return pd.DataFrame(rows)


def enrich_ou_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_injury_columns(frame)
    out["abs_close_move"] = _num(out, "close_move").abs()
    out["line_moved_toward_side"] = np.where(
        out["side"].eq("OVER"),
        _num(out, "close_move").gt(0),
        _num(out, "close_move").lt(0),
    ).astype(int)
    out["line_movement_magnitude"] = _num(out, "close_move").abs()
    out["expected_total_variance"] = (
        (_num(out, "home_game_total_l14") - _num(out, "home_game_total_l30")).abs()
        + (_num(out, "away_game_total_l14") - _num(out, "away_game_total_l30")).abs()
        + 0.25 * _num(out, "bullpen_fatigue_index", 0.0)
        + 0.10 * _num(out, "combined_barrel_rate", 7.5)
    ).fillna(0.0)
    out["team_plate_appearances_proxy"] = (
        4.2 * (_num(out, "home_runs_scored_l14") + _num(out, "away_runs_scored_l14")).fillna(9.0)
    ).clip(25, 95)
    out["umpire_run_factor"] = _num(out, "umpire_run_factor", 1.0).fillna(1.0)
    out["umpire_strike_zone_index"] = _num(out, "umpire_strike_zone_index", 0.0).fillna(0.0)
    for col in OU_EXTRA_NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def run_ou_v25(predictions: pd.DataFrame, features: pd.DataFrame | None, cfg: MarketIntelligenceV25Config) -> dict[str, pd.DataFrame]:
    class _OUCfg:
        min_open_edge = cfg.ou_min_open_edge
        max_open_edge = cfg.ou_max_open_edge

    frame = build_line_movement_frame(predictions, features, _OUCfg())
    frame = enrich_ou_frame(frame)
    numeric = [col for col in frame.columns if col in set(OU_EXTRA_NUMERIC_FEATURES)]
    from .ou_line_movement_model import BASE_NUMERIC_FEATURES, BASE_CATEGORICAL_FEATURES

    numeric_features = list(dict.fromkeys(BASE_NUMERIC_FEATURES + numeric))
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
    scored["rank_score"] = (scored["ou_clv_ev"] * np.sqrt(scored["ou_clv_probability"].clip(0.001, 0.999))).round(6)
    scored["profit_units"] = 0.0
    scored.loc[scored["game_result"].eq("WIN"), "profit_units"] = scored.loc[scored["game_result"].eq("WIN"), "payout_per_unit"]
    scored.loc[scored["game_result"].eq("LOSS"), "profit_units"] = -1.0
    gate_rows = []
    train_scored = scored[(scored["season"] >= cfg.train_start_season) & (scored["season"] <= cfg.train_end_season)]
    for (side, bucket), group in train_scored.groupby(["side", "opening_total_bucket"], sort=True):
        group = group[group["ou_clv_probability"].ge(cfg.ou_min_clv_probability) & group["ou_clv_ev"].ge(0)]
        row = _summary_row(group, f"{side}_{bucket}", result_col="game_result")
        row.update({"side": side, "opening_total_bucket": bucket, "allowed_segment": row["bets"] >= cfg.ou_min_segment_bets and row["clv_win_rate"] >= cfg.ou_min_segment_clv})
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
        & test["ou_clv_ev"].ge(0)
        & allowed_mask
    ].copy()
    selected = []
    for (_date, _side), group in pool.groupby(["game_date", "side"], sort=True):
        selected.append(group.sort_values(["rank_score", "ou_clv_ev", "open_edge"], ascending=[False, False, False]).head(cfg.ou_max_daily_per_side))
    pool = pd.concat(selected, ignore_index=False, sort=False) if selected else pd.DataFrame(columns=test.columns)
    daily = []
    for _, group in pool.groupby("game_date", sort=True):
        daily.append(group.sort_values(["rank_score", "ou_clv_ev", "open_edge"], ascending=[False, False, False]).head(cfg.ou_max_daily))
    orders = pd.concat(daily, ignore_index=True, sort=False) if daily else pd.DataFrame(columns=test.columns)
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
        "ou_overall": pd.DataFrame([_summary_row(orders, "ALL", result_col="game_result")]),
        "ou_segment_gate": gate,
        "ou_model_metrics": pd.DataFrame(metrics),
        "ou_model_metadata": pd.DataFrame(metadata),
    }


def run_v25(
    *,
    ml_scored_candidates: pd.DataFrame,
    ou_predictions: pd.DataFrame | None,
    features: pd.DataFrame | None,
    cfg: MarketIntelligenceV25Config,
) -> dict[str, pd.DataFrame]:
    ml_frame = prepare_ml_frame(ml_scored_candidates, cfg)
    ml_models, ml_metadata = fit_ml_clv_models(ml_frame, cfg)
    ml_scored = score_ml_clv(ml_frame, ml_models)
    ml_gate = build_ml_segment_gate(ml_scored, cfg)
    ml_orders = select_ml_orders(ml_scored, ml_gate, cfg)
    outputs = {
        "ml_scored_candidates": ml_scored,
        "ml_orders": ml_orders.reset_index(drop=True),
        "ml_overall": pd.DataFrame([_summary_row(ml_orders[ml_orders.get("execution_action", "").eq("BET")] if "execution_action" in ml_orders else ml_orders, "BET")]),
        "ml_observe_only": pd.DataFrame([_summary_row(ml_orders[ml_orders.get("execution_action", "").eq("OBSERVE_ONLY")] if "execution_action" in ml_orders else pd.DataFrame(), "OBSERVE_ONLY")]),
        "ml_segment_gate": ml_gate,
        "ml_model_metrics": ml_metrics(ml_scored, cfg),
        "ml_model_metadata": ml_metadata,
    }
    if ou_predictions is not None:
        outputs.update(run_ou_v25(ou_predictions, features, cfg))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLB v2.5 shared market intelligence: injury-aware ML/OU CLV selectors.")
    parser.add_argument("--ml-scored-candidates", required=True, type=Path)
    parser.add_argument("--ou-predictions", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-start-season", default=MarketIntelligenceV25Config.train_start_season, type=int)
    parser.add_argument("--train-end-season", default=MarketIntelligenceV25Config.train_end_season, type=int)
    parser.add_argument("--test-season", default=MarketIntelligenceV25Config.test_season, type=int)
    parser.add_argument("--ml-min-clv-probability", default=MarketIntelligenceV25Config.ml_min_clv_probability, type=float)
    parser.add_argument("--ou-min-clv-probability", default=MarketIntelligenceV25Config.ou_min_clv_probability, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MarketIntelligenceV25Config(
        train_start_season=args.train_start_season,
        train_end_season=args.train_end_season,
        test_season=args.test_season,
        ml_min_clv_probability=args.ml_min_clv_probability,
        ou_min_clv_probability=args.ou_min_clv_probability,
    )
    ml_scored = pd.read_csv(args.ml_scored_candidates, low_memory=False)
    ou_predictions = pd.read_csv(args.ou_predictions, low_memory=False) if args.ou_predictions else None
    features = pd.read_csv(args.features, low_memory=False) if args.features else None
    outputs = run_v25(ml_scored_candidates=ml_scored, ou_predictions=ou_predictions, features=features, cfg=cfg)
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
