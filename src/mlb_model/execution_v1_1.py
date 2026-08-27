from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .governance import canonical_hash, utc_now_iso, write_manifest
from .synthetic_totals import add_synthetic_total_line


EXECUTION_VERSION = "mlb_execution_v1_1"
QUALITY_WIDTH = 0.85


@dataclass(frozen=True)
class MLBExecutionV11Config:
    winner_enabled: bool = True
    market_ou_enabled: bool = True
    synthetic_ou_enabled: bool = True

    min_winner_margin: float = 1.25
    max_winner_margin: float = 6.0
    min_winner_probability: float = 0.535
    target_winner_margin: float = 2.0

    min_ou_edge: float = 0.75
    max_ou_edge: float = 8.0
    min_ou_probability: float = 0.525
    target_ou_edge: float = 1.5

    max_daily_winner_picks: int = 4
    max_daily_ou_picks: int = 4
    slate_pick_fraction: float = 0.30
    min_selection_score: float = 0.0
    min_batch_avg_probability: float = 0.535
    min_batch_avg_edge: float = 1.0

    stake_base: float = 1.0
    bankroll_units: float = 100.0
    max_stake_pct_bankroll: float = 0.01
    max_total_daily_exposure: float = 8.0
    max_market_daily_exposure: float = 4.0
    max_game_exposure: float = 1.0
    max_drawdown_pct: float = 0.15
    rolling_roi_throttle_threshold: float = -0.05
    rolling_roi_throttle_multiplier: float = 0.50
    previous_action_throttle_multiplier: float = 0.50
    global_kill_switch: bool = False

    synthetic_rolling_days: int = 30
    synthetic_min_games: int = 50
    synthetic_policy: Literal["observe_only", "bet"] = "observe_only"
    max_synthetic_market_gap: float = 2.0
    require_market_validation_for_synthetic: bool = False

    execution_version: str = EXECUTION_VERSION


def _append_reason(series: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    out = series.fillna("").copy()
    idx = pd.Series(mask, index=series.index).fillna(False).astype(bool)
    out.loc[idx] = out.loc[idx].where(out.loc[idx].eq(""), out.loc[idx] + "|") + reason
    return out


def _quality(edge_abs: pd.Series, target: float) -> pd.Series:
    return np.exp(-((edge_abs - target) / QUALITY_WIDTH) ** 2)


def _base_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "game_date", "game_pk", "pred_home_score", "pred_away_score", "pred_total"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"execution v1.1 predictions missing required columns: {sorted(missing)}")
    out = predictions.copy()
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="raise").astype(int)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise").dt.date.astype(str)
    for col in ["pred_home_score", "pred_away_score", "pred_total"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            raise ValueError(f"execution v1.1 predictions have invalid {col}")
    if "pred_margin" not in out.columns:
        out["pred_margin"] = out["pred_home_score"] - out["pred_away_score"]
    return out.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)


def _winner_candidates(predictions: pd.DataFrame, cfg: MLBExecutionV11Config) -> pd.DataFrame:
    out = predictions.copy()
    out["market"] = "WINNER"
    out["execution_mode"] = "winner"
    out["edge"] = pd.to_numeric(out["pred_margin"], errors="coerce").round(3)
    out["edge_abs"] = out["edge"].abs()
    out["side"] = np.where(out["edge"] > 0, "HOME", "AWAY")
    if {"home_team", "away_team"}.issubset(out.columns):
        out["display_side"] = out["home_team"].where(out["side"].eq("HOME"), out["away_team"])
    else:
        out["display_side"] = out["side"]
    out["heuristic_probability_score"] = (0.50 + (out["edge_abs"] * 0.03).clip(0, 0.18)).round(4)
    out["selection_quality"] = _quality(out["edge_abs"], cfg.target_winner_margin)
    out["selection_score"] = (out["heuristic_probability_score"] * (0.65 + 0.35 * out["selection_quality"])).round(6)
    out["execution_action"] = "ELIGIBLE"
    out["block_reason"] = ""
    out.loc[out["edge_abs"] < cfg.min_winner_margin, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], out["edge_abs"] < cfg.min_winner_margin, "MIN_WINNER_MARGIN")
    out.loc[out["edge_abs"] > cfg.max_winner_margin, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], out["edge_abs"] > cfg.max_winner_margin, "MAX_WINNER_MARGIN")
    out.loc[out["heuristic_probability_score"] < cfg.min_winner_probability, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], out["heuristic_probability_score"] < cfg.min_winner_probability, "MIN_PROB")
    return out


def _ou_candidates(predictions: pd.DataFrame, cfg: MLBExecutionV11Config, *, synthetic: bool) -> pd.DataFrame:
    out = predictions.copy()
    out["market"] = "OU"
    out["execution_mode"] = "synthetic_ou" if synthetic else "market_ou"
    if synthetic:
        out = add_synthetic_total_line(
            out,
            rolling_days=cfg.synthetic_rolling_days,
            min_games=cfg.synthetic_min_games,
        )
        if "market_total_line" not in out.columns and "total_line" in predictions.columns:
            out["market_total_line"] = pd.to_numeric(predictions["total_line"], errors="coerce")
    else:
        if "market_total_line" not in out.columns and "total_line" in out.columns:
            out["market_total_line"] = pd.to_numeric(out["total_line"], errors="coerce")

    if "total_line" not in out.columns:
        out["total_line"] = pd.NA
    out["total_line"] = pd.to_numeric(out["total_line"], errors="coerce")
    out["edge"] = (out["pred_total"] - out["total_line"]).round(3)
    out["edge_abs"] = out["edge"].abs()
    out["side"] = np.where(out["edge"] > 0, "OVER", "UNDER")
    out["display_side"] = out["side"]
    out["heuristic_probability_score"] = (0.50 + (out["edge_abs"] * 0.025).clip(0, 0.10)).round(4)
    out["selection_quality"] = _quality(out["edge_abs"], cfg.target_ou_edge)
    out["selection_score"] = (out["edge_abs"] * (0.50 + 0.50 * out["selection_quality"])).round(6)
    out["execution_action"] = "ELIGIBLE"
    out["block_reason"] = ""

    missing_line = out["total_line"].isna()
    out.loc[missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], missing_line, "MISSING_TOTAL_LINE")
    low_edge = out["edge_abs"] < cfg.min_ou_edge
    high_edge = out["edge_abs"] > cfg.max_ou_edge
    low_prob = out["heuristic_probability_score"] < cfg.min_ou_probability
    out.loc[low_edge & ~missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], low_edge & ~missing_line, "MIN_OU_EDGE")
    out.loc[high_edge & ~missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], high_edge & ~missing_line, "MAX_OU_EDGE")
    out.loc[low_prob & ~missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], low_prob & ~missing_line, "MIN_PROB")

    if synthetic:
        market_line = pd.to_numeric(out.get("market_total_line", pd.Series(pd.NA, index=out.index)), errors="coerce")
        out["synthetic_market_gap"] = (out["total_line"] - market_line).abs()
        missing_market = market_line.isna()
        too_far = out["synthetic_market_gap"] > cfg.max_synthetic_market_gap
        if cfg.require_market_validation_for_synthetic:
            out.loc[missing_market, "execution_action"] = "BLOCK"
            out["block_reason"] = _append_reason(out["block_reason"], missing_market, "SYNTHETIC_MISSING_MARKET_VALIDATION")
        out.loc[too_far & ~missing_market, "execution_action"] = "OBSERVE_ONLY"
        out["block_reason"] = _append_reason(out["block_reason"], too_far & ~missing_market, "SYNTHETIC_MARKET_DIVERGENCE")
        if cfg.synthetic_policy == "observe_only":
            active = out["execution_action"].eq("ELIGIBLE")
            out.loc[active, "execution_action"] = "OBSERVE_ONLY"
            out["block_reason"] = _append_reason(out["block_reason"], active, "SYNTHETIC_OBSERVE_ONLY")
    return out


def _grade(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if {"home_score", "away_score"}.issubset(out.columns):
        home_score = pd.to_numeric(out["home_score"], errors="coerce")
        away_score = pd.to_numeric(out["away_score"], errors="coerce")
        winner = out["market"].eq("WINNER")
        pred_home = out["side"].eq("HOME")
        actual_home = home_score > away_score
        out.loc[winner, "actual_result"] = np.where(pred_home[winner].eq(actual_home[winner]), "WIN", "LOSS")
    if {"total_runs", "total_line"}.issubset(out.columns):
        total_runs = pd.to_numeric(out["total_runs"], errors="coerce")
        total_line = pd.to_numeric(out["total_line"], errors="coerce")
        ou = out["market"].eq("OU")
        actual_over = total_runs > total_line
        actual_under = total_runs < total_line
        out.loc[ou, "actual_result"] = "PUSH"
        out.loc[ou & out["side"].eq("OVER") & actual_over, "actual_result"] = "WIN"
        out.loc[ou & out["side"].eq("OVER") & actual_under, "actual_result"] = "LOSS"
        out.loc[ou & out["side"].eq("UNDER") & actual_under, "actual_result"] = "WIN"
        out.loc[ou & out["side"].eq("UNDER") & actual_over, "actual_result"] = "LOSS"
    return out


def _apply_batch_quality(candidates: pd.DataFrame, cfg: MLBExecutionV11Config) -> pd.DataFrame:
    out = candidates.copy()
    for (mode, game_date), group in out[out["execution_action"].eq("ELIGIBLE")].groupby(["execution_mode", "game_date"], sort=False):
        avg_prob = group["heuristic_probability_score"].mean()
        avg_edge = group["edge_abs"].mean()
        weak = avg_prob < cfg.min_batch_avg_probability or avg_edge < cfg.min_batch_avg_edge
        if weak:
            idx = group.index
            out.loc[idx, "execution_action"] = "BLOCK"
            out["block_reason"] = _append_reason(out["block_reason"], out.index.isin(idx), "BATCH_QUALITY_GATE")
    return out


def _daily_cap(mode: str, slate_games: int, cfg: MLBExecutionV11Config) -> int:
    hard_cap = cfg.max_daily_winner_picks if mode == "winner" else cfg.max_daily_ou_picks
    adaptive = max(1, int(np.ceil(slate_games * cfg.slate_pick_fraction)))
    return min(hard_cap, adaptive)


def _stake_base(cfg: MLBExecutionV11Config, capital_state: dict | None) -> float:
    bankroll = cfg.bankroll_units
    if isinstance(capital_state, dict):
        try:
            bankroll = float(capital_state.get("bankroll_units", bankroll))
        except (TypeError, ValueError):
            bankroll = cfg.bankroll_units
    return min(cfg.stake_base, bankroll * cfg.max_stake_pct_bankroll)


def _risk_multiplier(cfg: MLBExecutionV11Config, risk_state: dict | None, previous_action: str | None) -> tuple[float, list[str], bool]:
    multiplier = 1.0
    reasons: list[str] = []
    kill = cfg.global_kill_switch
    if previous_action and str(previous_action).upper() in {"THROTTLE", "RECOVERY", "PAUSE"}:
        multiplier *= cfg.previous_action_throttle_multiplier
        reasons.append("PREVIOUS_ACTION_THROTTLE")
    if isinstance(risk_state, dict):
        try:
            drawdown = float(risk_state.get("current_drawdown_pct", 0.0))
            if drawdown >= cfg.max_drawdown_pct:
                kill = True
                reasons.append("MAX_DRAWDOWN_KILL")
        except (TypeError, ValueError):
            reasons.append("INVALID_DRAWDOWN")
        try:
            rolling_roi = float(risk_state.get("rolling_roi", 0.0))
            if rolling_roi < cfg.rolling_roi_throttle_threshold:
                multiplier *= cfg.rolling_roi_throttle_multiplier
                reasons.append("ROLLING_ROI_THROTTLE")
        except (TypeError, ValueError):
            reasons.append("INVALID_ROLLING_ROI")
    if cfg.global_kill_switch:
        reasons.append("GLOBAL_KILL_SWITCH")
    return multiplier, reasons, kill


def _select_orders(
    candidates: pd.DataFrame,
    cfg: MLBExecutionV11Config,
    *,
    capital_state: dict | None,
    risk_state: dict | None,
    previous_action: str | None,
) -> pd.DataFrame:
    out = _apply_batch_quality(candidates, cfg)
    out["selected_rank"] = pd.NA
    out["stake_base"] = _stake_base(cfg, capital_state)
    out["stake_final"] = 0.0
    out["order_intent"] = "NO_ORDER"
    out["throttle_reason"] = ""
    multiplier, risk_reasons, kill = _risk_multiplier(cfg, risk_state, previous_action)

    if kill:
        mask = out["execution_action"].isin(["ELIGIBLE", "OBSERVE_ONLY"])
        out.loc[mask, "execution_action"] = "BLOCK"
        for reason in risk_reasons:
            out["block_reason"] = _append_reason(out["block_reason"], mask, reason)
        return out

    selected_indexes: list[int] = []
    eligible = out[out["execution_action"].eq("ELIGIBLE")]
    for (game_date, mode), group in eligible.groupby(["game_date", "execution_mode"], sort=True):
        slate_games = int(out[out["game_date"].eq(game_date)]["game_pk"].nunique())
        cap = _daily_cap(mode, slate_games, cfg)
        ordered = group.sort_values(
            ["selection_score", "heuristic_probability_score", "edge_abs", "game_pk", "side"],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        )
        chosen = ordered.head(cap)
        selected_indexes.extend(chosen.index.tolist())
        out.loc[chosen.index, "selected_rank"] = range(1, len(chosen) + 1)

    if selected_indexes:
        out.loc[selected_indexes, "execution_action"] = "BET"
    cutoff = out["execution_action"].eq("ELIGIBLE")
    out.loc[cutoff, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], cutoff, "ADAPTIVE_RANK_CUTOFF")

    bet_mask = out["execution_action"].eq("BET")
    if multiplier < 1.0:
        out.loc[bet_mask, "execution_action"] = "THROTTLE"
        out.loc[bet_mask, "throttle_reason"] = "|".join(risk_reasons)
    wager_mask = out["execution_action"].isin(["BET", "THROTTLE"])
    out.loc[wager_mask, "stake_final"] = out.loc[wager_mask, "stake_base"] * multiplier
    out.loc[out["execution_action"].eq("BET"), "order_intent"] = "NORMAL_ORDER"
    out.loc[out["execution_action"].eq("THROTTLE"), "order_intent"] = "REDUCED_STAKE_ORDER"
    out.loc[out["execution_action"].eq("OBSERVE_ONLY"), "order_intent"] = "OBSERVE_ONLY"

    out = _trim_exposure(out, ["game_date", "game_pk"], cfg.max_game_exposure, "GAME_EXPOSURE_TRIM")
    out = _trim_exposure(out, ["game_date", "market"], cfg.max_market_daily_exposure, "MARKET_DAILY_EXPOSURE_TRIM")
    out = _trim_exposure(out, ["game_date"], cfg.max_total_daily_exposure, "TOTAL_DAILY_EXPOSURE_TRIM")
    return out


def _trim_exposure(out: pd.DataFrame, keys: list[str], cap: float, reason: str) -> pd.DataFrame:
    if cap <= 0:
        mask = out["execution_action"].isin(["BET", "THROTTLE"])
        out.loc[mask, "execution_action"] = "BLOCK"
        out["block_reason"] = _append_reason(out["block_reason"], mask, reason)
        out.loc[mask, "stake_final"] = 0.0
        out.loc[mask, "order_intent"] = "NO_ORDER"
        return out
    wager = out[out["execution_action"].isin(["BET", "THROTTLE"])]
    for _key, group in wager.groupby(keys, sort=False):
        ordered = group.sort_values(
            ["selection_score", "heuristic_probability_score", "edge_abs", "game_pk", "side"],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        )
        running = ordered["stake_final"].cumsum()
        drop_idx = ordered.index[running > cap]
        if len(drop_idx):
            mask = out.index.isin(drop_idx)
            out.loc[mask, "execution_action"] = "BLOCK"
            out["block_reason"] = _append_reason(out["block_reason"], mask, reason)
            out.loc[mask, "stake_final"] = 0.0
            out.loc[mask, "order_intent"] = "NO_ORDER"
    return out


def _summarize(frame: pd.DataFrame, period: Literal["overall", "daily", "weekly"]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    group_cols = ["execution_mode", "market"]
    if period == "daily":
        group_cols.append("game_date")
    if period == "weekly":
        dates = pd.to_datetime(out["game_date"], errors="raise")
        iso = dates.dt.isocalendar()
        out["test_week"] = iso["week"].astype(int)
        out["week_start"] = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.date.astype(str)
        group_cols.extend(["test_week", "week_start"])
    rows = []
    for keys, group in out.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        wagers = group[group["execution_action"].isin(["BET", "THROTTLE"])]
        row.update(
            {
                "games": int(len(group)),
                "orders": int(len(wagers)),
                "observe_only": int(group["execution_action"].eq("OBSERVE_ONLY").sum()),
                "blocked": int(group["execution_action"].eq("BLOCK").sum()),
                "stake_units": float(wagers["stake_final"].sum()) if "stake_final" in wagers else 0.0,
                "avg_edge_abs": float(wagers["edge_abs"].mean()) if len(wagers) else 0.0,
                "avg_probability": float(wagers["heuristic_probability_score"].mean()) if len(wagers) else 0.0,
                "wins": int(wagers["actual_result"].eq("WIN").sum()) if "actual_result" in wagers else 0,
                "losses": int(wagers["actual_result"].eq("LOSS").sum()) if "actual_result" in wagers else 0,
                "pushes": int(wagers["actual_result"].eq("PUSH").sum()) if "actual_result" in wagers else 0,
            }
        )
        row["profit_units"] = 0.0
        if "actual_result" in wagers:
            row["profit_units"] = float(
                (wagers["actual_result"].eq("WIN") * wagers["stake_final"] * 0.9091).sum()
                - (wagers["actual_result"].eq("LOSS") * wagers["stake_final"]).sum()
            )
        decisions = row["wins"] + row["losses"]
        row["win_rate"] = float(row["wins"] / decisions) if decisions else 0.0
        row["roi"] = float(row["profit_units"] / row["stake_units"]) if row["stake_units"] else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def execute_v1_1(
    predictions: pd.DataFrame,
    *,
    config: MLBExecutionV11Config | None = None,
    capital_state: dict | None = None,
    risk_state: dict | None = None,
    previous_action: str | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or MLBExecutionV11Config()
    base = _base_predictions(predictions)
    parts: list[pd.DataFrame] = []
    synthetic_predictions = pd.DataFrame()
    if cfg.winner_enabled:
        parts.append(_winner_candidates(base, cfg))
    if cfg.market_ou_enabled:
        parts.append(_ou_candidates(base, cfg, synthetic=False))
    if cfg.synthetic_ou_enabled:
        synthetic_predictions = add_synthetic_total_line(
            base,
            rolling_days=cfg.synthetic_rolling_days,
            min_games=cfg.synthetic_min_games,
        )
        synth_input = base.copy()
        if "total_line" in synth_input.columns:
            synth_input["market_total_line"] = pd.to_numeric(synth_input["total_line"], errors="coerce")
        parts.append(_ou_candidates(synth_input, cfg, synthetic=True))
    candidates = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    candidates = _grade(candidates)
    candidates["execution_version"] = EXECUTION_VERSION
    candidates["input_hash"] = canonical_hash(predictions)
    candidates["config_hash"] = canonical_hash(asdict(cfg))
    selected = _select_orders(
        candidates,
        cfg,
        capital_state=capital_state,
        risk_state=risk_state,
        previous_action=previous_action,
    )
    selected["generated_at_utc"] = utc_now_iso()
    selected = selected.sort_values(
        ["game_date", "execution_mode", "execution_action", "selection_score", "game_pk", "side"],
        ascending=[True, True, True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    deterministic = selected.drop(columns=["generated_at_utc", "execution_hash"], errors="ignore")
    selected["execution_hash"] = canonical_hash(deterministic)
    orders = selected[selected["execution_action"].isin(["BET", "THROTTLE"])].copy().reset_index(drop=True)
    return {
        "orders": orders,
        "audit_candidates": selected,
        "daily_summary": _summarize(selected, "daily"),
        "weekly_summary": _summarize(selected, "weekly"),
        "overall_summary": _summarize(selected, "overall"),
        "predictions_with_synthetic_line": synthetic_predictions,
    }


def _load_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run governed MLB execution v1.1.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--capital-state", type=Path)
    parser.add_argument("--risk-state", type=Path)
    parser.add_argument("--previous-action")
    parser.add_argument("--winner-daily-cap", default=MLBExecutionV11Config.max_daily_winner_picks, type=int)
    parser.add_argument("--ou-daily-cap", default=MLBExecutionV11Config.max_daily_ou_picks, type=int)
    parser.add_argument("--slate-pick-fraction", default=MLBExecutionV11Config.slate_pick_fraction, type=float)
    parser.add_argument("--synthetic-policy", choices=["observe_only", "bet"], default=MLBExecutionV11Config.synthetic_policy)
    parser.add_argument("--require-market-validation-for-synthetic", action="store_true")
    parser.add_argument("--max-synthetic-market-gap", default=MLBExecutionV11Config.max_synthetic_market_gap, type=float)
    parser.add_argument("--max-game-exposure", default=MLBExecutionV11Config.max_game_exposure, type=float)
    parser.add_argument("--max-market-daily-exposure", default=MLBExecutionV11Config.max_market_daily_exposure, type=float)
    parser.add_argument("--max-total-daily-exposure", default=MLBExecutionV11Config.max_total_daily_exposure, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    cfg = MLBExecutionV11Config(
        max_daily_winner_picks=args.winner_daily_cap,
        max_daily_ou_picks=args.ou_daily_cap,
        slate_pick_fraction=args.slate_pick_fraction,
        synthetic_policy=args.synthetic_policy,
        require_market_validation_for_synthetic=args.require_market_validation_for_synthetic,
        max_synthetic_market_gap=args.max_synthetic_market_gap,
        max_game_exposure=args.max_game_exposure,
        max_market_daily_exposure=args.max_market_daily_exposure,
        max_total_daily_exposure=args.max_total_daily_exposure,
    )
    outputs = execute_v1_1(
        predictions,
        config=cfg,
        capital_state=_load_json(args.capital_state),
        risk_state=_load_json(args.risk_state),
        previous_action=args.previous_action,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "execution_version": EXECUTION_VERSION,
            "generated_at_utc": utc_now_iso(),
            "predictions_hash": canonical_hash(predictions),
            "config": asdict(cfg),
            "capital_state_hash": canonical_hash(_load_json(args.capital_state)) if args.capital_state else None,
            "risk_state_hash": canonical_hash(_load_json(args.risk_state)) if args.risk_state else None,
            "previous_action": args.previous_action,
            "outputs": {name: canonical_hash(frame) for name, frame in outputs.items()},
        },
    )
    print(outputs["overall_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
