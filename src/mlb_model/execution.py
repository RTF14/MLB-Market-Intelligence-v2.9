from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd

from .config import MLBExecutionConfig
from .governance import canonical_hash, utc_now_iso


BET_COLUMNS = [
    "season",
    "game_date",
    "game_pk",
    "market",
    "side",
    "sportsbook_side",
    "execution_action",
    "block_reason",
    "home_team",
    "away_team",
    "total_line",
    "pred_total",
    "pred_home_score",
    "pred_away_score",
    "pred_edge",
    "edge_abs",
    "selection_quality",
    "selection_score",
    "heuristic_probability_score",
    "stake_base",
    "stake_final",
    "execution_version",
    "input_hash",
    "config_hash",
    "execution_hash",
    "order_intent",
    "generated_at_utc",
]


def _append_reason(series: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    out = series.fillna("").copy()
    idx = pd.Series(mask, index=series.index).fillna(False).astype(bool)
    out.loc[idx] = out.loc[idx].where(out.loc[idx].eq(""), out.loc[idx] + "|") + reason
    return out


def _quality(edge: pd.Series, target: float) -> pd.Series:
    return np.exp(-((edge.abs() - target) / 0.75) ** 2)


def validate_prediction_frame(preds: pd.DataFrame, *, cfg: MLBExecutionConfig) -> pd.DataFrame:
    required = {"season", "game_date", "game_pk", "pred_total", "pred_home_score", "pred_away_score"}
    if cfg.require_total_line:
        required.add("total_line")
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"MLB execution input missing required columns: {sorted(missing)}")

    out = preds.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise").dt.date.astype(str)
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="raise").astype(int)
    for col in ["pred_total", "pred_home_score", "pred_away_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any() or (out[col] < 0).any():
            raise ValueError(f"MLB execution input has invalid {col}")
    if "total_line" in out.columns:
        out["total_line"] = pd.to_numeric(out["total_line"], errors="coerce")
        if out["total_line"].isna().any():
            raise ValueError("MLB execution input has invalid total_line")
    if cfg.require_unique_games and out.duplicated(["game_pk"]).any():
        examples = out.loc[out.duplicated(["game_pk"], keep=False), ["game_pk", "game_date"]].head(5)
        raise ValueError(f"MLB execution input has duplicate games: {examples.to_dict('records')}")
    return out.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def execute_mlb_ou(
    preds: pd.DataFrame,
    *,
    config: Optional[MLBExecutionConfig] = None,
    previous_action: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or MLBExecutionConfig()
    df = validate_prediction_frame(preds, cfg=cfg)
    input_hash = canonical_hash(df)
    config_hash = canonical_hash(asdict(cfg))

    out = df.copy()
    out["market"] = cfg.market
    out["pred_edge"] = (out["pred_total"] - out["total_line"]).round(2)
    out["edge_abs"] = out["pred_edge"].abs()
    out["side"] = np.where(out["pred_edge"] > 0, "OVER", "UNDER")
    out["sportsbook_side"] = out["side"] + " " + out["total_line"].map(lambda x: f"{x:g}")
    out["selection_quality"] = _quality(out["pred_edge"], cfg.target_edge)
    out["selection_score"] = (out["edge_abs"] * (0.50 + 0.50 * out["selection_quality"])).round(4)
    out["heuristic_probability_score"] = (0.50 + (out["edge_abs"] * 0.025).clip(0, 0.10)).round(4)
    out["execution_action"] = "ELIGIBLE"
    out["block_reason"] = ""

    below_edge = out["edge_abs"] < cfg.min_abs_edge
    above_edge = out["edge_abs"] > cfg.max_abs_edge
    low_prob = out["heuristic_probability_score"] < cfg.min_model_probability
    out.loc[below_edge, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], below_edge, "MIN_EDGE")
    out.loc[above_edge, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], above_edge, "MAX_EDGE_GUARDRAIL")
    out.loc[low_prob, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], low_prob, "MIN_PROB")
    if previous_action and str(previous_action).upper() in {"PAUSE", "KILL", "BLOCK"}:
        mask = pd.Series(True, index=out.index)
        out.loc[mask, "execution_action"] = "BLOCK"
        out["block_reason"] = _append_reason(out["block_reason"], mask, "PREVIOUS_ACTION_BLOCK")
    if cfg.global_kill_switch:
        out["execution_action"] = "BLOCK"
        out["block_reason"] = "KILL_SWITCH"

    selected_parts = []
    eligible = out[out["execution_action"].eq("ELIGIBLE")].copy()
    for (_season, _date), group in eligible.groupby(["season", "game_date"], sort=False):
        chosen = group.sort_values(["selection_score", "edge_abs", "game_pk"], ascending=[False, False, True]).head(cfg.daily_cap)
        selected_parts.append(chosen)
    if selected_parts:
        selected_idx = pd.concat(selected_parts).index
        out.loc[selected_idx, "execution_action"] = "BET"
    still_eligible = out["execution_action"].eq("ELIGIBLE")
    out.loc[still_eligible, "execution_action"] = "BLOCK"
    out.loc[still_eligible, "block_reason"] = "DAILY_CAP_CUTOFF"

    stake_base = min(cfg.stake_base, cfg.bankroll_units * cfg.max_stake_pct_bankroll)
    out["stake_base"] = stake_base
    out["stake_final"] = np.where(out["execution_action"].eq("BET"), stake_base, 0.0)
    for (_season, _date), group in out[out["execution_action"].eq("BET")].groupby(["season", "game_date"], sort=False):
        ordered = group.sort_values(["selection_score", "edge_abs"], ascending=[False, False])
        drop_idx = ordered.index[ordered["stake_final"].cumsum() > cfg.max_total_exposure]
        out.loc[drop_idx, "execution_action"] = "BLOCK"
        out["block_reason"] = _append_reason(out["block_reason"], out.index.isin(drop_idx), "TOTAL_EXPOSURE_TRIM")
        out.loc[drop_idx, "stake_final"] = 0.0

    out["order_intent"] = "NO_ORDER"
    out.loc[out["execution_action"].eq("BET"), "order_intent"] = "NORMAL_ORDER"
    out["execution_version"] = cfg.execution_version
    out["input_hash"] = input_hash
    out["config_hash"] = config_hash
    out["generated_at_utc"] = utc_now_iso()
    deterministic = out.drop(columns=["execution_hash", "generated_at_utc"], errors="ignore")
    out["execution_hash"] = canonical_hash(deterministic)

    cols = [col for col in BET_COLUMNS if col in out.columns]
    audit = out[cols + [col for col in out.columns if col not in cols]].sort_values(
        ["season", "game_date", "execution_action", "selection_score"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    orders = audit[audit["execution_action"].eq("BET")].copy().reset_index(drop=True)
    return {"orders": orders, "audit_candidates": audit, "bets": orders}
