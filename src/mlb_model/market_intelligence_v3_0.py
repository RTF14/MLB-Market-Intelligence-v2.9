from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .governance import canonical_hash, utc_now_iso, write_manifest
from .market_intelligence_v2_5 import build_pipeline
from .market_intelligence_v2_7 import _allowed_ml_segment_mask, _ml_model_metrics, _ml_thresholds, _ou_model_metrics, _shrunk_values
from .market_intelligence_v2_8 import (
    _merge_stake_multiplier,
    _pitcher_key,
    build_ml_segment_gate_v28,
    fit_group_models_v28,
    prepare_ml_frame_v28,
    score_ml_v28,
)
from .market_intelligence_v2_8b import (
    OU_CATEGORICAL_FEATURES_V28B,
    OU_PRUNING_CANDIDATES,
    MarketIntelligenceV28BConfig,
    _available_numeric,
    _ridge_pipeline,
    _select_ou_features,
    enrich_ou_frame_v28b,
)
from .market_snapshots import add_clv_snapshot_fields, build_snapshot_tables
from .ou_line_movement_model import build_line_movement_frame


RESEARCH_VERSION = "mlb_market_intelligence_v3_0"


@dataclass(frozen=True)
class MarketIntelligenceV30Config(MarketIntelligenceV28BConfig):
    ml_soft_daily_target: int = 2
    ou_soft_daily_target: int = 2
    ml_backfill_edge_multiplier: float = 0.90
    ml_backfill_min_ev_open: float = 0.03
    ml_backfill_min_game_probability: float = 0.52
    ml_backfill_min_clv_probability: float = 0.535
    ou_backfill_expected_close_delta: float = 0.15
    backfill_stake_multiplier: float = 0.50
    min_stake_multiplier: float = 0.50
    max_stake_multiplier: float = 1.50
    ev_stake_weight: float = 2.0
    research_version: str = RESEARCH_VERSION


def _unit_profit(df: pd.DataFrame, result_col: str) -> pd.Series:
    profit = pd.Series(0.0, index=df.index, dtype=float)
    if result_col not in df.columns:
        return profit
    payout = pd.to_numeric(df.get("payout_per_unit", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    profit.loc[df[result_col].eq("WIN")] = payout.loc[df[result_col].eq("WIN")]
    profit.loc[df[result_col].eq("LOSS")] = -1.0
    return profit


def _apply_stake_scaling(df: pd.DataFrame, cfg: MarketIntelligenceV30Config, *, market: str, result_col: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["stake_units"] = pd.Series(dtype=float)
        out["unit_profit"] = pd.Series(dtype=float)
        return out
    ev_col = "ml_risk_adjusted_ev" if market == "ML" and "ml_risk_adjusted_ev" in out.columns else "ou_game_ev"
    ev = pd.to_numeric(out.get(ev_col, pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).clip(lower=0.0)
    tier_scale = np.where(out.get("selection_tier", pd.Series("STRICT", index=out.index)).astype(str).eq("BACKFILL"), cfg.backfill_stake_multiplier, 1.0)
    segment_scale = pd.to_numeric(out.get("stake_multiplier", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0)
    out["stake_units"] = (tier_scale * segment_scale * (1.0 + cfg.ev_stake_weight * ev)).clip(cfg.min_stake_multiplier, cfg.max_stake_multiplier)
    out["unit_profit"] = _unit_profit(out, result_col)
    out["profit_units"] = out["unit_profit"] * out["stake_units"]
    return out


def _summary_row_v30(df: pd.DataFrame, label: str, result_col: str = "actual_result") -> dict:
    wins = int(df[result_col].eq("WIN").sum()) if result_col in df else 0
    losses = int(df[result_col].eq("LOSS").sum()) if result_col in df else 0
    pushes = int(df[result_col].eq("PUSH").sum()) if result_col in df else 0
    decisions = wins + losses
    stake = float(pd.to_numeric(df.get("stake_units", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0).sum()) if len(df) else 0.0
    profit = float(pd.to_numeric(df.get("profit_units", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0).sum()) if len(df) else 0.0
    clv_decisions = df["clv_result"].isin(["WIN", "LOSS"]) if "clv_result" in df and len(df) else pd.Series(dtype=bool)
    return {
        "segment": label,
        "bets": int(len(df)),
        "stake_units": stake,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": wins / decisions if decisions else 0.0,
        "profit_units": profit,
        "roi": profit / stake if stake else 0.0,
        "clv_wins": int(df["clv_result"].eq("WIN").sum()) if "clv_result" in df and len(df) else 0,
        "clv_losses": int(df["clv_result"].eq("LOSS").sum()) if "clv_result" in df and len(df) else 0,
        "clv_win_rate": float(df.loc[clv_decisions, "clv_result"].eq("WIN").mean()) if len(clv_decisions) and clv_decisions.any() else 0.0,
    }


def _tier_summary(df: pd.DataFrame, *, market: str, result_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{"market": market, **_summary_row_v30(df, "ALL", result_col)}])
    rows = [{"market": market, **_summary_row_v30(df, "ALL", result_col)}]
    for tier, group in df.groupby("selection_tier", sort=True):
        rows.append({"market": market, **_summary_row_v30(group, str(tier), result_col)})
    return pd.DataFrame(rows)


def _adaptive_daily_select(
    strict: pd.DataFrame,
    backfill: pd.DataFrame,
    *,
    soft_target: int,
    max_daily: int,
    sort_cols: list[str],
    ascending: list[bool],
    max_team_exposure: int | None = None,
    max_pitcher_exposure: int | None = None,
    max_per_side: int | None = None,
) -> pd.DataFrame:
    selected = []
    dates = sorted(set(strict.get("game_date", pd.Series(dtype=object)).dropna().astype(str)) | set(backfill.get("game_date", pd.Series(dtype=object)).dropna().astype(str)))
    for game_date in dates:
        day_strict = strict[strict["game_date"].astype(str).eq(game_date)].copy()
        day_backfill = backfill[backfill["game_date"].astype(str).eq(game_date)].copy()
        chosen = []
        used_games: set[str] = set()
        used_teams: dict[str, int] = {}
        used_pitchers: dict[str, int] = {}
        used_sides: dict[str, int] = {}

        def add_from(pool: pd.DataFrame, target: int, tier: str) -> None:
            if pool.empty:
                return
            for idx, row in pool.sort_values(sort_cols, ascending=ascending, kind="mergesort").iterrows():
                if len(chosen) >= target or len(chosen) >= max_daily:
                    break
                game_key = str(row.get("game_pk", ""))
                if game_key in used_games:
                    continue
                side = str(row.get("side", ""))
                if max_per_side is not None and side and used_sides.get(side, 0) >= max_per_side:
                    continue
                teams = [str(row.get("home_team", "")), str(row.get("away_team", ""))]
                if max_team_exposure is not None and any(used_teams.get(team, 0) >= max_team_exposure for team in teams if team):
                    continue
                pitcher = str(row.get("pitcher_exposure_key", ""))
                if max_pitcher_exposure is not None and pitcher and pitcher.lower() != "nan" and used_pitchers.get(pitcher, 0) >= max_pitcher_exposure:
                    continue
                row = row.copy()
                row["selection_tier"] = tier
                chosen.append(row)
                if game_key:
                    used_games.add(game_key)
                if side:
                    used_sides[side] = used_sides.get(side, 0) + 1
                for team in teams:
                    if team:
                        used_teams[team] = used_teams.get(team, 0) + 1
                if pitcher and pitcher.lower() != "nan":
                    used_pitchers[pitcher] = used_pitchers.get(pitcher, 0) + 1

        add_from(day_strict, max_daily, "STRICT")
        # Soft target: backfill can top up a day that already has signal, but
        # a no-strict day stays empty rather than manufacturing action.
        if 0 < len(chosen) < soft_target:
            day_backfill = day_backfill[~day_backfill["game_pk"].astype(str).isin(used_games)]
            add_from(day_backfill, soft_target, "BACKFILL")
        if chosen:
            selected.append(pd.DataFrame(chosen))
    return pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame(columns=strict.columns)


def run_ml_v30(ml_scored_candidates: pd.DataFrame, cfg: MarketIntelligenceV30Config) -> dict[str, pd.DataFrame]:
    frame = prepare_ml_frame_v28(ml_scored_candidates, cfg)
    clv_models, clv_metadata = fit_group_models_v28(frame, cfg, "target_clv_win")
    game_models, game_metadata = fit_group_models_v28(frame, cfg, "target_game_win")
    scored = score_ml_v28(frame, clv_models, game_models, cfg)
    gate = build_ml_segment_gate_v28(scored, cfg)
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    edge_threshold, ev_threshold = _ml_thresholds(test, cfg)
    allowed = _allowed_ml_segment_mask(test, gate)
    common = test["model_probability"].ge(cfg.ml_min_win_probability) & test["ml_game_ev_open"].ge(0) & test["ml_clv_ev"].ge(0) & allowed
    strict = test[common & test["ml_game_probability"].ge(cfg.ml_min_game_probability) & test["probability_edge"].ge(edge_threshold) & test["ev_open"].ge(ev_threshold) & test["ml_clv_probability"].ge(cfg.ml_min_clv_probability)].copy()
    backfill = test[common & test["ml_game_probability"].ge(cfg.ml_backfill_min_game_probability) & test["probability_edge"].ge(edge_threshold * cfg.ml_backfill_edge_multiplier) & test["ev_open"].ge(cfg.ml_backfill_min_ev_open) & test["ml_clv_probability"].ge(cfg.ml_backfill_min_clv_probability)].copy()
    for pool in [strict, backfill]:
        if not pool.empty:
            pool["pitcher_exposure_key"] = pool.apply(_pitcher_key, axis=1)
    orders = _adaptive_daily_select(
        strict,
        backfill,
        soft_target=cfg.ml_soft_daily_target,
        max_daily=cfg.ml_max_daily,
        sort_cols=["rank_score", "ml_risk_adjusted_ev", "ml_game_ev_open", "game_pk"],
        ascending=[False, False, False, True],
        max_team_exposure=cfg.max_daily_team_exposure,
        max_pitcher_exposure=cfg.max_daily_pitcher_exposure,
    )
    orders = _merge_stake_multiplier(orders, gate, ["ml_side_segment", "price_band"])
    orders = orders.sort_values(["game_date", "rank_score", "game_pk"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    orders["execution_action"] = "BET"
    orders["market"] = "ML"
    orders = add_clv_snapshot_fields(orders, market="ML", mode=cfg.snapshot_mode)
    orders = _apply_stake_scaling(orders, cfg, market="ML", result_col="actual_result")
    bet_snapshots, closing_snapshots = build_snapshot_tables(orders)
    return {
        "ml_scored_candidates": scored,
        "ml_orders": orders.reset_index(drop=True),
        "ml_bet_snapshots": bet_snapshots,
        "ml_closing_snapshots": closing_snapshots,
        "ml_overall": pd.DataFrame([_summary_row_v30(orders, "BET")]),
        "ml_tier_summary": _tier_summary(orders, market="ML", result_col="actual_result"),
        "ml_segment_gate": gate,
        "ml_model_metrics": pd.concat([_ml_model_metrics(scored, cfg, "ml_clv_probability", "target_clv_win", "clv"), _ml_model_metrics(scored, cfg, "ml_game_probability", "target_game_win", "game")], ignore_index=True),
        "ml_model_metadata": pd.concat([clv_metadata, game_metadata], ignore_index=True),
    }


def run_ou_v30(predictions: pd.DataFrame, features: pd.DataFrame | None, cfg: MarketIntelligenceV30Config) -> dict[str, pd.DataFrame]:
    class _OUCfg:
        min_open_edge = cfg.ou_min_open_edge
        max_open_edge = cfg.ou_max_open_edge

    frame = enrich_ou_frame_v28b(build_line_movement_frame(predictions, features, _OUCfg()))
    numeric_pool = _available_numeric(frame, OU_PRUNING_CANDIDATES)
    train = frame[(frame["season"] >= cfg.train_start_season) & (frame["season"] <= cfg.train_end_season)].copy()
    delta_models = {}
    game_models = {}
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
        metadata.append({"market": "OU", "model_group": side, "training_rows": int(len(group)), "target": "expected_close_delta_for_side + target_game_win", "selected_feature_count": int(len(selected_features)), "selected_features": "|".join(selected_features), "training_hash": canonical_hash(group)})

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
    scored["rank_score"] = (scored["expected_close_delta_for_side"] * scored["ou_game_ev"]).round(6)

    gate_rows = []
    train_scored = scored[(scored["season"] >= cfg.train_start_season) & (scored["season"] <= cfg.train_end_season)]
    for (side, bucket), group in train_scored.groupby(["side", "total_bucket_group"], sort=True):
        eligible = group[group["expected_close_delta_for_side"].ge(cfg.ou_backfill_expected_close_delta) & group["ou_game_probability"].ge(cfg.ou_min_game_probability) & group["ou_game_ev"].ge(cfg.ou_min_game_ev)]
        row = _summary_row_v30(eligible, f"{side}_{bucket}", result_col="game_result")
        shrunk_clv, shrunk_roi = _shrunk_values(eligible, cfg.ou_segment_prior_bets, cfg.ou_segment_prior_clv)
        row.update({"side": side, "total_bucket_group": bucket, "shrunk_clv_win_rate": shrunk_clv, "shrunk_roi": shrunk_roi})
        row["allowed_segment"] = row["bets"] >= cfg.ou_min_segment_bets and shrunk_clv >= cfg.ou_segment_min_shrunk_clv and shrunk_roi >= cfg.ou_segment_min_shrunk_roi
        gate_rows.append(row)
    gate = pd.DataFrame(gate_rows)

    test = scored[scored["season"].eq(cfg.test_season)].copy()
    if gate.empty:
        allowed = pd.Series(False, index=test.index)
    else:
        allowed_gate = gate[gate["allowed_segment"].astype(bool)]
        keys = set(zip(allowed_gate["side"].astype(str), allowed_gate["total_bucket_group"].astype(str)))
        allowed = pd.Series(list(zip(test["side"].astype(str), test["total_bucket_group"].astype(str))), index=test.index).isin(keys)
    common = test["ou_game_probability"].ge(cfg.ou_min_game_probability) & test["ou_game_ev"].ge(cfg.ou_min_game_ev) & allowed
    strict = test[common & test["expected_close_delta_for_side"].ge(cfg.ou_min_expected_close_delta)].copy()
    backfill = test[common & test["expected_close_delta_for_side"].ge(cfg.ou_backfill_expected_close_delta)].copy()
    orders = _adaptive_daily_select(
        strict,
        backfill,
        soft_target=cfg.ou_soft_daily_target,
        max_daily=cfg.ou_max_daily,
        sort_cols=["rank_score", "expected_close_delta_for_side", "ou_game_ev", "game_pk", "side"],
        ascending=[False, False, False, True, True],
        max_per_side=cfg.ou_max_daily_per_side,
    )
    orders["market"] = "OU"
    orders = add_clv_snapshot_fields(orders, market="OU", mode=cfg.snapshot_mode)
    orders = _apply_stake_scaling(orders, cfg, market="OU", result_col="game_result")
    bet_snapshots, closing_snapshots = build_snapshot_tables(orders)
    return {
        "ou_scored_candidates": scored,
        "ou_orders": orders.reset_index(drop=True),
        "ou_bet_snapshots": bet_snapshots,
        "ou_closing_snapshots": closing_snapshots,
        "ou_overall": pd.DataFrame([_summary_row_v30(orders, "ALL", result_col="game_result")]),
        "ou_tier_summary": _tier_summary(orders, market="OU", result_col="game_result"),
        "ou_segment_gate": gate,
        "ou_feature_ablation": pd.concat(feature_ranks, ignore_index=True) if feature_ranks else pd.DataFrame(),
        "ou_model_metrics": _ou_model_metrics(scored, cfg, "ou_game_probability", "target_game_win", "game"),
        "ou_model_metadata": pd.DataFrame(metadata),
    }


def run_v30(*, ml_scored_candidates: pd.DataFrame, ou_predictions: pd.DataFrame | None, features: pd.DataFrame | None, cfg: MarketIntelligenceV30Config) -> dict[str, pd.DataFrame]:
    outputs = run_ml_v30(ml_scored_candidates, cfg)
    if ou_predictions is not None:
        outputs.update(run_ou_v30(ou_predictions, features, cfg))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLB v3.0 adaptive edge selector with tier reporting and stake scaling.")
    parser.add_argument("--ml-scored-candidates", required=True, type=Path)
    parser.add_argument("--ou-predictions", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-start-season", default=MarketIntelligenceV30Config.train_start_season, type=int)
    parser.add_argument("--train-end-season", default=MarketIntelligenceV30Config.train_end_season, type=int)
    parser.add_argument("--test-season", default=MarketIntelligenceV30Config.test_season, type=int)
    parser.add_argument("--snapshot-mode", choices=["historical_backtest", "live_paper"], default=MarketIntelligenceV30Config.snapshot_mode)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MarketIntelligenceV30Config(train_start_season=args.train_start_season, train_end_season=args.train_end_season, test_season=args.test_season, snapshot_mode=args.snapshot_mode)
    ml_scored = pd.read_csv(args.ml_scored_candidates, low_memory=False)
    ou_predictions = pd.read_csv(args.ou_predictions, low_memory=False) if args.ou_predictions else None
    features = pd.read_csv(args.features, low_memory=False) if args.features else None
    outputs = run_v30(ml_scored_candidates=ml_scored, ou_predictions=ou_predictions, features=features, cfg=cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(args.out_dir / "manifest.json", {"research_version": RESEARCH_VERSION, "generated_at_utc": utc_now_iso(), "ml_scored_hash": canonical_hash(ml_scored), "ou_predictions_hash": canonical_hash(ou_predictions) if ou_predictions is not None else None, "features_hash": canonical_hash(features) if features is not None else None, "config": asdict(cfg), "outputs": {name: canonical_hash(frame) for name, frame in outputs.items()}})
    print(outputs["ml_overall"].to_string(index=False))
    if "ou_overall" in outputs:
        print(outputs["ou_overall"].to_string(index=False))


if __name__ == "__main__":
    main()
