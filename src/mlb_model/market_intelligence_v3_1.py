from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .market_intelligence_v3_0_strict_rc1 import MarketIntelligenceV30StrictRC1Config


RESEARCH_VERSION = "mlb_market_intelligence_v3_1"


@dataclass(frozen=True)
class MarketIntelligenceV31Config(MarketIntelligenceV30StrictRC1Config):
    ml_weekly_cap: int = 3
    ml_weekly_soft_min: int = 2
    allow_weekly_ml_select: bool = True
    weekly_ml_min_game_probability: float = 0.58
    weekly_ml_probability_column: str = "model_probability"
    weekly_ml_min_probability_edge: float = 0.04
    weekly_ml_min_ev: float = 0.05
    weekly_ml_require_positive_clv_ev: bool = True
    weekly_ml_block_check_odds: bool = True

    all_ml_probability_column: str = "model_probability"
    all_ml_min_game_probability: float = 0.56
    all_ml_min_ev: float = 0.0
    all_ml_min_probability_edge: float = 0.0
    all_ou_min_game_probability: float = 0.56
    all_ou_min_open_edge: float = 2.5
    all_ou_min_ev: float = 0.03
    research_version: str = RESEARCH_VERSION


def _iso_week_start(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="raise")
    return (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.date.astype(str)


def _is_check_odds(df: pd.DataFrame) -> pd.Series:
    if "needs_cross_reference" in df.columns:
        return df["needs_cross_reference"].fillna(False).astype(bool)
    return pd.Series(False, index=df.index)


def ml_score_agrees(row: pd.Series) -> bool:
    margin = pd.to_numeric(pd.Series([row.get("model_margin_for_side")]), errors="coerce").iloc[0]
    return bool(pd.notna(margin) and margin > 0)


def add_weekly_ml_select(
    *,
    ml_scored_candidates: pd.DataFrame,
    strict_ml_orders: pd.DataFrame,
    cfg: MarketIntelligenceV31Config,
) -> pd.DataFrame:
    if not cfg.allow_weekly_ml_select or ml_scored_candidates.empty:
        return pd.DataFrame(columns=ml_scored_candidates.columns)

    test = ml_scored_candidates[ml_scored_candidates["season"].eq(cfg.test_season)].copy()
    if test.empty:
        return pd.DataFrame(columns=ml_scored_candidates.columns)
    test["week_start"] = _iso_week_start(test["game_date"])

    strict = strict_ml_orders.copy()
    if strict.empty:
        strict = pd.DataFrame(columns=test.columns)
    if not strict.empty:
        strict["week_start"] = _iso_week_start(strict["game_date"])

    used_games = set(strict.get("game_pk", pd.Series(dtype=object)).astype(str))
    checks = _is_check_odds(test)
    probability_col = cfg.weekly_ml_probability_column if cfg.weekly_ml_probability_column in test.columns else "ml_game_probability"
    eligible = (
        test[probability_col].ge(cfg.weekly_ml_min_game_probability)
        & test["probability_edge"].ge(cfg.weekly_ml_min_probability_edge)
        & test["ml_game_ev_open"].ge(cfg.weekly_ml_min_ev)
        & test.apply(ml_score_agrees, axis=1)
        & ~test["game_pk"].astype(str).isin(used_games)
    )
    if cfg.weekly_ml_require_positive_clv_ev and "ml_clv_ev" in test.columns:
        eligible &= test["ml_clv_ev"].ge(0)
    if cfg.weekly_ml_block_check_odds:
        eligible &= ~checks

    pool = test[eligible].copy()
    if pool.empty:
        return pool

    selected = []
    all_weeks = sorted(set(test["week_start"].dropna().astype(str)))
    for week in all_weeks:
        week_strict_count = int(strict["week_start"].astype(str).eq(week).sum()) if not strict.empty else 0
        slots = max(cfg.ml_weekly_cap - week_strict_count, 0)
        if slots <= 0:
            continue
        # Soft-min behavior: top up only to 2 when strict is light, but never exceed cap.
        target = cfg.ml_weekly_soft_min - week_strict_count
        target = max(min(target, slots), 0)
        if target <= 0:
            continue
        week_pool = pool[pool["week_start"].astype(str).eq(week)].copy()
        if week_pool.empty:
            continue
        game_best = (
            week_pool.sort_values(
                [probability_col, "ml_game_ev_open", "rank_score", "game_pk", "side"],
                ascending=[False, False, False, True, True],
                kind="mergesort",
            )
            .groupby("game_pk")
            .head(1)
        )
        chosen = game_best.head(target).copy()
        if chosen.empty:
            continue
        chosen["selection_tier"] = "WEEKLY_SELECT"
        chosen["execution_action"] = "BET"
        chosen["market"] = "ML"
        chosen["stake_units"] = 1.0
        chosen["profit_units"] = np.where(
            chosen["actual_result"].eq("WIN"),
            pd.to_numeric(chosen["payout_per_unit"], errors="coerce").fillna(0.0),
            np.where(chosen["actual_result"].eq("LOSS"), -1.0, 0.0),
        )
        selected.append(chosen)

    if not selected:
        return pd.DataFrame(columns=test.columns)
    return pd.concat(selected, ignore_index=True, sort=False)


def cap_strict_ml_by_week(strict_ml_orders: pd.DataFrame, cfg: MarketIntelligenceV31Config) -> pd.DataFrame:
    if strict_ml_orders.empty:
        return strict_ml_orders.copy()
    out = strict_ml_orders.copy()
    out["week_start"] = _iso_week_start(out["game_date"])
    parts = []
    for _week, group in out.groupby("week_start", sort=True):
        parts.append(
            group.sort_values(
                ["rank_score", "ml_risk_adjusted_ev", "ml_game_ev_open", "game_pk"],
                ascending=[False, False, False, True],
                kind="mergesort",
            ).head(cfg.ml_weekly_cap)
        )
    capped = pd.concat(parts, ignore_index=True, sort=False) if parts else out.iloc[0:0].copy()
    capped["weekly_cap_policy"] = f"MAX_{cfg.ml_weekly_cap}_ML_PER_WEEK_STRICT_FIRST"
    return capped


def select_all_ml_with_no_pick(scored: pd.DataFrame, cfg: MarketIntelligenceV31Config) -> pd.DataFrame:
    test = scored[scored["season"].eq(cfg.test_season)].copy()
    if test.empty:
        return test
    checks = _is_check_odds(test)
    probability_col = cfg.all_ml_probability_column if cfg.all_ml_probability_column in test.columns else "ml_game_probability"
    eligible = (
        test[probability_col].ge(cfg.all_ml_min_game_probability)
        & test["ml_game_ev_open"].ge(cfg.all_ml_min_ev)
        & test["probability_edge"].ge(cfg.all_ml_min_probability_edge)
        & test.apply(ml_score_agrees, axis=1)
        & ~checks
    )
    pool = test[eligible].copy()
    if pool.empty:
        return pd.DataFrame(columns=list(test.columns) + ["pick_status"])
    picks = (
        pool.sort_values([probability_col, "ml_game_ev_open", "rank_score", "game_pk"], ascending=[False, False, False, True], kind="mergesort")
        .groupby("game_pk")
        .head(1)
        .copy()
    )
    picks["pick_status"] = "PICK"
    picks["stake_units"] = 1.0
    picks["profit_units"] = np.where(
        picks["actual_result"].eq("WIN"),
        pd.to_numeric(picks["payout_per_unit"], errors="coerce").fillna(0.0),
        np.where(picks["actual_result"].eq("LOSS"), -1.0, 0.0),
    )
    return picks


def select_all_ou_with_no_pick(ou_scored: pd.DataFrame, cfg: MarketIntelligenceV31Config) -> pd.DataFrame:
    test = ou_scored[ou_scored["season"].eq(cfg.test_season)].copy()
    if test.empty:
        return test
    checks = _is_check_odds(test)
    eligible = (
        test["ou_game_probability"].ge(cfg.all_ou_min_game_probability)
        & pd.to_numeric(test["open_edge"], errors="coerce").abs().ge(cfg.all_ou_min_open_edge)
        & test["ou_game_ev"].ge(cfg.all_ou_min_ev)
        & ~checks
    )
    picks = test[eligible].copy()
    if picks.empty:
        return pd.DataFrame(columns=list(test.columns) + ["pick_status"])
    picks = (
        picks.sort_values(["ou_game_probability", "ou_game_ev", "rank_score", "game_pk"], ascending=[False, False, False, True], kind="mergesort")
        .groupby("game_pk")
        .head(1)
        .copy()
    )
    picks["pick_status"] = "PICK"
    picks["stake_units"] = 1.0
    payout = pd.to_numeric(picks.get("payout_per_unit", pd.Series(0.0, index=picks.index)), errors="coerce").fillna(0.0)
    picks["profit_units"] = np.where(picks["game_result"].eq("WIN"), payout, np.where(picks["game_result"].eq("LOSS"), -1.0, 0.0))
    return picks
