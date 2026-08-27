from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from .governance import canonical_hash, utc_now_iso, write_manifest
from .market_intelligence_v2_5 import _num, _summary_row, build_pipeline
from .market_intelligence_v2_6 import _normal_cdf, enrich_ou_frame_v26
from .market_intelligence_v2_7 import MarketIntelligenceV27Config, _ou_model_metrics, _shrunk_values, _total_bucket_group
from .market_intelligence_v2_8 import MarketIntelligenceV28Config, run_ml_v28
from .market_snapshots import add_clv_snapshot_fields, build_snapshot_tables
from .ou_line_movement_model import build_line_movement_frame


RESEARCH_VERSION = "mlb_market_intelligence_v2_8b"


@dataclass(frozen=True)
class MarketIntelligenceV28BConfig(MarketIntelligenceV28Config):
    ou_max_selected_features: int = 20
    ou_min_expected_close_delta: float = 0.20
    ou_min_game_probability: float = 0.515
    ou_min_game_ev: float = 0.0
    ou_segment_min_shrunk_clv: float = 0.52
    ou_segment_min_shrunk_roi: float = 0.0
    research_version: str = RESEARCH_VERSION


OU_PRUNING_CANDIDATES = [
    # Core market/edge signal.
    "model_total",
    "opening_total",
    "model_minus_open",
    "abs_model_minus_open",
    "opening_total_bucket",
    "open_vig",
    "over_open_implied_probability",
    "under_open_implied_probability",
    # Pitcher.
    "home_sp_xFIP",
    "away_sp_xFIP",
    "home_sp_kbb",
    "away_sp_kbb",
    "starter_xFIP_diff",
    "starter_kbb_diff",
    "home_starter_recent_form_index",
    "away_starter_recent_form_index",
    "starter_recent_form_diff",
    # Bullpen.
    "home_bullpen_ip_proxy_l1d",
    "away_bullpen_ip_proxy_l1d",
    "home_bullpen_ip_proxy_l3d",
    "away_bullpen_ip_proxy_l3d",
    "home_bullpen_fatigue_rate_l3d",
    "away_bullpen_fatigue_rate_l3d",
    "bullpen_fatigue_index",
    "bullpen_fatigue_diff",
    "bullpen_xFIP_diff",
    # Weather / park / total buckets.
    "temperature_f",
    "wind_mph",
    "humidity_pct",
    "weather_run_index",
    "park_run_factor",
    # Offense context kept intentionally compact.
    "home_wRC_plus_vs_hand",
    "away_wRC_plus_vs_hand",
    "offense_wRC_plus_diff",
    "combined_barrel_rate",
    "team_offense_form_diff_l7",
    "team_offense_form_diff_l14",
    "team_offense_form_diff_l30",
    "home_game_total_l7",
    "away_game_total_l7",
]

OU_CATEGORICAL_FEATURES_V28B = ["side", "venue_id", "total_bucket_group"]


def _available_numeric(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [col for col in candidates if col in frame.columns]


def _side_close_delta(frame: pd.DataFrame) -> pd.Series:
    sign = np.where(frame["side"].astype(str).eq("OVER"), 1.0, -1.0)
    return pd.Series(sign * _num(frame, "close_move", 0.0), index=frame.index, dtype=float)


def _feature_rank(train: pd.DataFrame, features: list[str], target: str, side: str) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(train[target], errors="coerce")
    for feature in features:
        x = pd.to_numeric(train[feature], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 20 or x.loc[valid].nunique() <= 1:
            score = 0.0
        else:
            score = abs(float(x.loc[valid].corr(y.loc[valid])))
            if np.isnan(score):
                score = 0.0
        rows.append({"side": side, "feature": feature, "target": target, "abs_corr": score, "non_null_rows": int(valid.sum())})
    return pd.DataFrame(rows).sort_values(["abs_corr", "feature"], ascending=[False, True], kind="mergesort")


def _select_ou_features(train: pd.DataFrame, features: list[str], cfg: MarketIntelligenceV28BConfig, side: str) -> tuple[list[str], pd.DataFrame]:
    rank = _feature_rank(train, features, "target_close_delta_for_side", side)
    selected = rank.head(cfg.ou_max_selected_features)["feature"].tolist()
    if "model_minus_open" in features and "model_minus_open" not in selected:
        selected = ["model_minus_open"] + selected[: max(cfg.ou_max_selected_features - 1, 0)]
    return selected, rank.assign(selected=lambda df: df["feature"].isin(selected))


def _ridge_pipeline(numeric_features: list[str], categorical_features: list[str]):
    # Reuse the repo's preprocessing style, but swap the final classifier for a
    # small ridge regressor so OU optimizes expected close-line delta directly.
    pipe = build_pipeline(numeric_features, categorical_features)
    pipe.steps[-1] = ("model", Ridge(alpha=2.0))
    return pipe


def enrich_ou_frame_v28b(frame: pd.DataFrame) -> pd.DataFrame:
    out = enrich_ou_frame_v26(frame)
    out["target_game_win"] = out["game_result"].eq("WIN").astype(int)
    out["target_close_delta_for_side"] = _side_close_delta(out)
    out["total_bucket_group"] = _total_bucket_group(out["opening_total"]).astype(str)
    out["ou_result_probability_formula"] = _normal_cdf(_num(out, "open_edge") / np.sqrt((2.25 + _num(out, "expected_total_variance", 0.0)).clip(1.0, 25.0)))
    out["profit_units"] = 0.0
    out.loc[out["game_result"].eq("WIN"), "profit_units"] = out.loc[out["game_result"].eq("WIN"), "payout_per_unit"]
    out.loc[out["game_result"].eq("LOSS"), "profit_units"] = -1.0
    for col in OU_PRUNING_CANDIDATES:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in OU_CATEGORICAL_FEATURES_V28B:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str)
    return out.sort_values(["game_date", "game_pk", "side"], kind="mergesort").reset_index(drop=True)


def run_ou_v28b(predictions: pd.DataFrame, features: pd.DataFrame | None, cfg: MarketIntelligenceV28BConfig) -> dict[str, pd.DataFrame]:
    class _OUCfg:
        min_open_edge = cfg.ou_min_open_edge
        max_open_edge = cfg.ou_max_open_edge

    frame = enrich_ou_frame_v28b(build_line_movement_frame(predictions, features, _OUCfg()))
    numeric_pool = _available_numeric(frame, OU_PRUNING_CANDIDATES)
    train = frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)].copy()

    delta_models: dict[str, object] = {}
    game_models: dict[str, object] = {}
    metadata = []
    feature_ranks = []
    for side, group in train.groupby("side", sort=True):
        selected_features, rank = _select_ou_features(group, numeric_pool, cfg, side)
        feature_ranks.append(rank)
        if group["target_game_win"].nunique() < 2:
            continue
        delta_model = _ridge_pipeline(selected_features, OU_CATEGORICAL_FEATURES_V28B)
        game_model = build_pipeline(selected_features, OU_CATEGORICAL_FEATURES_V28B)
        delta_model.fit(group[selected_features + OU_CATEGORICAL_FEATURES_V28B], group["target_close_delta_for_side"])
        game_model.fit(group[selected_features + OU_CATEGORICAL_FEATURES_V28B], group["target_game_win"].astype(int))
        delta_models[side] = (delta_model, selected_features)
        game_models[side] = (game_model, selected_features)
        metadata.append(
            {
                "market": "OU",
                "model_group": side,
                "training_rows": int(len(group)),
                "target": "expected_close_delta_for_side + target_game_win",
                "selected_feature_count": int(len(selected_features)),
                "selected_features": "|".join(selected_features),
                "training_hash": canonical_hash(group),
                "sklearn_version": sklearn.__version__,
            }
        )

    scored = frame.copy()
    scored["expected_close_delta_for_side"] = np.nan
    scored["ou_game_probability"] = np.nan
    for side, (model, selected_features) in delta_models.items():
        mask = scored["side"].eq(side)
        if mask.any():
            scored.loc[mask, "expected_close_delta_for_side"] = model.predict(scored.loc[mask, selected_features + OU_CATEGORICAL_FEATURES_V28B])
    for side, (model, selected_features) in game_models.items():
        mask = scored["side"].eq(side)
        if mask.any():
            scored.loc[mask, "ou_game_probability"] = model.predict_proba(scored.loc[mask, selected_features + OU_CATEGORICAL_FEATURES_V28B])[:, 1]

    scored["ou_game_ev"] = scored["ou_game_probability"] * scored["payout_per_unit"] - (1.0 - scored["ou_game_probability"])
    scored["rank_score"] = scored["expected_close_delta_for_side"].round(6)

    gate_rows = []
    train_scored = scored[(scored["season"] >= cfg.train_start_season) & (scored["season"] <= cfg.train_end_season)]
    for (side, bucket), group in train_scored.groupby(["side", "total_bucket_group"], sort=True):
        eligible = group[
            group["expected_close_delta_for_side"].ge(cfg.ou_min_expected_close_delta)
            & group["ou_game_probability"].ge(cfg.ou_min_game_probability)
            & group["ou_game_ev"].ge(cfg.ou_min_game_ev)
        ]
        row = _summary_row(eligible, f"{side}_{bucket}", result_col="game_result")
        shrunk_clv, shrunk_roi = _shrunk_values(eligible, cfg.ou_segment_prior_bets, cfg.ou_segment_prior_clv)
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
        test["expected_close_delta_for_side"].ge(cfg.ou_min_expected_close_delta)
        & test["ou_game_probability"].ge(cfg.ou_min_game_probability)
        & test["ou_game_ev"].ge(cfg.ou_min_game_ev)
        & allowed_mask
    ].copy()

    selected = []
    for (_date, _side), group in pool.groupby(["game_date", "side"], sort=True):
        selected.append(group.sort_values(["rank_score", "ou_game_ev", "open_edge", "game_pk"], ascending=[False, False, False, True], kind="mergesort").head(cfg.ou_max_daily_per_side))
    pool = pd.concat(selected, ignore_index=False, sort=False) if selected else pd.DataFrame(columns=test.columns)
    daily = [group.sort_values(["rank_score", "ou_game_ev", "open_edge", "game_pk", "side"], ascending=[False, False, False, True, True], kind="mergesort").head(cfg.ou_max_daily) for _, group in pool.groupby("game_date", sort=True)]
    orders = pd.concat(daily, ignore_index=True, sort=False) if daily else pd.DataFrame(columns=test.columns)
    orders["market"] = "OU"
    orders = add_clv_snapshot_fields(orders, market="OU", mode=cfg.snapshot_mode)
    bet_snapshots, closing_snapshots = build_snapshot_tables(orders)

    metrics = []
    test_scored = scored[scored["season"].eq(cfg.test_season)].copy()
    for side, group in test_scored.groupby("side", sort=True):
        row = {"market": "OU", "model_group": side, "model": "closing_delta", "rows": int(len(group)), "target": "target_close_delta_for_side"}
        valid = group["expected_close_delta_for_side"].notna()
        if valid.any():
            row["mae"] = mean_absolute_error(group.loc[valid, "target_close_delta_for_side"], group.loc[valid, "expected_close_delta_for_side"])
            row["rmse"] = float(np.sqrt(mean_squared_error(group.loc[valid, "target_close_delta_for_side"], group.loc[valid, "expected_close_delta_for_side"])))
        metrics.append(row)
    game_metrics = _ou_model_metrics(scored, cfg, "ou_game_probability", "target_game_win", "game")
    return {
        "ou_scored_candidates": scored,
        "ou_orders": orders.reset_index(drop=True),
        "ou_bet_snapshots": bet_snapshots,
        "ou_closing_snapshots": closing_snapshots,
        "ou_overall": pd.DataFrame([_summary_row(orders, "ALL", result_col="game_result")]),
        "ou_segment_gate": gate,
        "ou_feature_ablation": pd.concat(feature_ranks, ignore_index=True) if feature_ranks else pd.DataFrame(),
        "ou_model_metrics": pd.concat([pd.DataFrame(metrics), game_metrics], ignore_index=True, sort=False),
        "ou_model_metadata": pd.DataFrame(metadata),
    }


def run_v28b(*, ml_scored_candidates: pd.DataFrame, ou_predictions: pd.DataFrame | None, features: pd.DataFrame | None, cfg: MarketIntelligenceV28BConfig) -> dict[str, pd.DataFrame]:
    outputs = run_ml_v28(ml_scored_candidates, cfg)
    if ou_predictions is not None:
        outputs.update(run_ou_v28b(ou_predictions, features, cfg))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLB v2.8b minimal high-signal OU with v2.8 ML.")
    parser.add_argument("--ml-scored-candidates", required=True, type=Path)
    parser.add_argument("--ou-predictions", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-start-season", default=MarketIntelligenceV28BConfig.train_start_season, type=int)
    parser.add_argument("--train-end-season", default=MarketIntelligenceV28BConfig.train_end_season, type=int)
    parser.add_argument("--test-season", default=MarketIntelligenceV28BConfig.test_season, type=int)
    parser.add_argument("--snapshot-mode", choices=["historical_backtest", "live_paper"], default=MarketIntelligenceV28BConfig.snapshot_mode)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MarketIntelligenceV28BConfig(train_start_season=args.train_start_season, train_end_season=args.train_end_season, test_season=args.test_season, snapshot_mode=args.snapshot_mode)
    ml_scored = pd.read_csv(args.ml_scored_candidates, low_memory=False)
    ou_predictions = pd.read_csv(args.ou_predictions, low_memory=False) if args.ou_predictions else None
    features = pd.read_csv(args.features, low_memory=False) if args.features else None
    outputs = run_v28b(ml_scored_candidates=ml_scored, ou_predictions=ou_predictions, features=features, cfg=cfg)
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
