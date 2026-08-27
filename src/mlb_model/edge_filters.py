from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUEdgeConfig, WinnerEdgeConfig
from .governance import canonical_hash, utc_now_iso, write_manifest


COMMON_COLUMNS = [
    "season",
    "game_date",
    "game_pk",
    "market",
    "side",
    "execution_action",
    "block_reason",
    "away_team",
    "home_team",
    "pred_home_score",
    "pred_away_score",
    "pred_total",
    "pred_margin",
    "edge",
    "edge_abs",
    "heuristic_probability_score",
    "selection_score",
    "stake_final",
    "actual_result",
    "profit_units",
]


def _append_reason(series: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    out = series.fillna("").copy()
    idx = pd.Series(mask, index=series.index).fillna(False).astype(bool)
    out.loc[idx] = out.loc[idx].where(out.loc[idx].eq(""), out.loc[idx] + "|") + reason
    return out


def _quality(edge_abs: pd.Series, target: float) -> pd.Series:
    return np.exp(-((edge_abs - target) / 0.85) ** 2)


def _base_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "game_date", "game_pk", "pred_home_score", "pred_away_score", "pred_total"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {sorted(missing)}")
    out = predictions.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise").dt.date.astype(str)
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="raise").astype(int)
    for col in ["pred_home_score", "pred_away_score", "pred_total"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            raise ValueError(f"Predictions have null/non-numeric {col}")
    if "pred_margin" not in out.columns:
        out["pred_margin"] = out["pred_home_score"] - out["pred_away_score"]
    return out.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def _select_daily(out: pd.DataFrame, daily_cap: int) -> pd.DataFrame:
    selected = []
    eligible = out[out["execution_action"].eq("ELIGIBLE")]
    for (_season, _date), group in eligible.groupby(["season", "game_date"], sort=False):
        selected.append(
            group.sort_values(["selection_score", "edge_abs", "game_pk"], ascending=[False, False, True]).head(daily_cap)
        )
    if selected:
        out.loc[pd.concat(selected).index, "execution_action"] = "BET"
    still_eligible = out["execution_action"].eq("ELIGIBLE")
    out.loc[still_eligible, "execution_action"] = "BLOCK"
    out.loc[still_eligible, "block_reason"] = "DAILY_CAP_CUTOFF"
    return out


def filter_winner_edges(
    predictions: pd.DataFrame,
    *,
    config: WinnerEdgeConfig | None = None,
) -> pd.DataFrame:
    cfg = config or WinnerEdgeConfig()
    out = _base_frame(predictions)
    out["market"] = "WINNER"
    out["edge"] = out["pred_margin"].round(3)
    out["edge_abs"] = out["edge"].abs()
    out["side"] = np.where(out["edge"] > 0, "HOME", "AWAY")
    out["pred_winner"] = out["home_team"].where(out["side"].eq("HOME"), out["away_team"]) if {"home_team", "away_team"}.issubset(out.columns) else out["side"]
    out["heuristic_probability_score"] = (0.50 + (out["edge_abs"] * 0.03).clip(0, 0.18)).round(4)
    out["selection_score"] = (out["heuristic_probability_score"] * (0.65 + 0.35 * _quality(out["edge_abs"], cfg.target_margin))).round(4)
    out["execution_action"] = "ELIGIBLE"
    out["block_reason"] = ""
    out["execution_version"] = cfg.execution_version
    out["input_hash"] = canonical_hash(predictions)
    out["config_hash"] = canonical_hash(asdict(cfg))

    below = out["edge_abs"] < cfg.min_abs_margin
    above = out["edge_abs"] > cfg.max_abs_margin
    low_prob = out["heuristic_probability_score"] < cfg.min_model_probability
    out.loc[below, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], below, "MIN_MARGIN_EDGE")
    out.loc[above, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], above, "MAX_MARGIN_GUARDRAIL")
    out.loc[low_prob, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], low_prob, "MIN_PROB")
    out = _select_daily(out, cfg.daily_cap)

    stake = min(cfg.stake_base, cfg.bankroll_units * cfg.max_stake_pct_bankroll)
    out["stake_final"] = np.where(out["execution_action"].eq("BET"), stake, 0.0)
    if {"home_score", "away_score"}.issubset(out.columns):
        actual_home = pd.to_numeric(out["home_score"], errors="coerce") > pd.to_numeric(out["away_score"], errors="coerce")
        pred_home = out["side"].eq("HOME")
        out["actual_result"] = np.where(actual_home.eq(pred_home), "WIN", "LOSS")
        out.loc[pd.to_numeric(out["home_score"], errors="coerce").eq(pd.to_numeric(out["away_score"], errors="coerce")), "actual_result"] = "PUSH"
        out["profit_units"] = 0.0
        out.loc[out["actual_result"].eq("WIN") & out["execution_action"].eq("BET"), "profit_units"] = out["stake_final"] * 0.9091
        out.loc[out["actual_result"].eq("LOSS") & out["execution_action"].eq("BET"), "profit_units"] = -out["stake_final"]
    out["generated_at_utc"] = utc_now_iso()
    out["execution_hash"] = canonical_hash(out.drop(columns=["generated_at_utc", "execution_hash"], errors="ignore"))
    return _ordered(out)


def filter_ou_edges(
    predictions: pd.DataFrame,
    *,
    config: OUEdgeConfig | None = None,
) -> pd.DataFrame:
    cfg = config or OUEdgeConfig()
    out = _base_frame(predictions)
    out["market"] = "OU"
    out["execution_action"] = "ELIGIBLE"
    out["block_reason"] = ""
    out["execution_version"] = cfg.execution_version
    out["input_hash"] = canonical_hash(predictions)
    out["config_hash"] = canonical_hash(asdict(cfg))

    if "total_line" not in out.columns:
        out["total_line"] = pd.NA
    out["total_line"] = pd.to_numeric(out["total_line"], errors="coerce")
    out["edge"] = (out["pred_total"] - out["total_line"]).round(3)
    out["edge_abs"] = out["edge"].abs()
    out["side"] = np.where(out["edge"] > 0, "OVER", "UNDER")
    out["heuristic_probability_score"] = (0.50 + (out["edge_abs"] * 0.025).clip(0, 0.10)).round(4)
    out["selection_score"] = (out["edge_abs"] * (0.50 + 0.50 * _quality(out["edge_abs"], cfg.target_total_edge))).round(4)

    missing_line = out["total_line"].isna()
    below = out["edge_abs"] < cfg.min_abs_total_edge
    above = out["edge_abs"] > cfg.max_abs_total_edge
    low_prob = out["heuristic_probability_score"] < cfg.min_model_probability
    out.loc[missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], missing_line, "MISSING_TOTAL_LINE")
    out.loc[below & ~missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], below & ~missing_line, "MIN_TOTAL_EDGE")
    out.loc[above & ~missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], above & ~missing_line, "MAX_TOTAL_EDGE_GUARDRAIL")
    out.loc[low_prob & ~missing_line, "execution_action"] = "BLOCK"
    out["block_reason"] = _append_reason(out["block_reason"], low_prob & ~missing_line, "MIN_PROB")
    out = _select_daily(out, cfg.daily_cap)

    stake = min(cfg.stake_base, cfg.bankroll_units * cfg.max_stake_pct_bankroll)
    out["stake_final"] = np.where(out["execution_action"].eq("BET"), stake, 0.0)
    if {"total_runs", "total_line"}.issubset(out.columns):
        actual_over = pd.to_numeric(out["total_runs"], errors="coerce") > out["total_line"]
        actual_under = pd.to_numeric(out["total_runs"], errors="coerce") < out["total_line"]
        out["actual_result"] = "PUSH"
        out.loc[out["side"].eq("OVER") & actual_over, "actual_result"] = "WIN"
        out.loc[out["side"].eq("OVER") & actual_under, "actual_result"] = "LOSS"
        out.loc[out["side"].eq("UNDER") & actual_under, "actual_result"] = "WIN"
        out.loc[out["side"].eq("UNDER") & actual_over, "actual_result"] = "LOSS"
        out["profit_units"] = 0.0
        out.loc[out["actual_result"].eq("WIN") & out["execution_action"].eq("BET"), "profit_units"] = out["stake_final"] * 0.9091
        out.loc[out["actual_result"].eq("LOSS") & out["execution_action"].eq("BET"), "profit_units"] = -out["stake_final"]
    out["generated_at_utc"] = utc_now_iso()
    out["execution_hash"] = canonical_hash(out.drop(columns=["generated_at_utc", "execution_hash"], errors="ignore"))
    return _ordered(out)


def summarize_daily(filtered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (market, game_date), group in filtered.groupby(["market", "game_date"], sort=True):
        bets = group[group["execution_action"].eq("BET")]
        row = {
            "market": market,
            "game_date": game_date,
            "games": int(len(group)),
            "picks": int(len(bets)),
            "avg_edge_abs": float(bets["edge_abs"].mean()) if len(bets) else 0.0,
            "avg_probability": float(bets["heuristic_probability_score"].mean()) if len(bets) else 0.0,
            "stake_units": float(bets["stake_final"].sum()) if "stake_final" in bets else 0.0,
        }
        if "actual_result" in bets:
            row["wins"] = int(bets["actual_result"].eq("WIN").sum())
            row["losses"] = int(bets["actual_result"].eq("LOSS").sum())
            row["pushes"] = int(bets["actual_result"].eq("PUSH").sum())
            row["profit_units"] = float(bets["profit_units"].sum()) if "profit_units" in bets else 0.0
            row["win_rate"] = float(row["wins"] / (row["wins"] + row["losses"])) if row["wins"] + row["losses"] else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_weekly(filtered: pd.DataFrame) -> pd.DataFrame:
    frame = filtered.copy()
    dates = pd.to_datetime(frame["game_date"], errors="raise")
    iso = dates.dt.isocalendar()
    frame["test_week"] = iso["week"].astype(int)
    frame["week_start"] = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.date.astype(str)
    rows = []
    for (market, test_week, week_start), group in frame.groupby(["market", "test_week", "week_start"], sort=True):
        bets = group[group["execution_action"].eq("BET")]
        row = {
            "market": market,
            "test_week": int(test_week),
            "week_start": week_start,
            "games": int(len(group)),
            "picks": int(len(bets)),
            "avg_edge_abs": float(bets["edge_abs"].mean()) if len(bets) else 0.0,
            "avg_probability": float(bets["heuristic_probability_score"].mean()) if len(bets) else 0.0,
            "stake_units": float(bets["stake_final"].sum()) if "stake_final" in bets else 0.0,
        }
        if "actual_result" in bets:
            row["wins"] = int(bets["actual_result"].eq("WIN").sum())
            row["losses"] = int(bets["actual_result"].eq("LOSS").sum())
            row["pushes"] = int(bets["actual_result"].eq("PUSH").sum())
            row["profit_units"] = float(bets["profit_units"].sum()) if "profit_units" in bets else 0.0
            row["win_pct"] = float(row["wins"] / (row["wins"] + row["losses"])) if row["wins"] + row["losses"] else 0.0
            row["roi"] = float(row["profit_units"] / row["stake_units"]) if row["stake_units"] else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_overall(filtered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, group in filtered.groupby("market", sort=True):
        bets = group[group["execution_action"].eq("BET")]
        row = {
            "market": market,
            "games": int(len(group)),
            "picks": int(len(bets)),
            "avg_edge_abs": float(bets["edge_abs"].mean()) if len(bets) else 0.0,
            "avg_probability": float(bets["heuristic_probability_score"].mean()) if len(bets) else 0.0,
            "stake_units": float(bets["stake_final"].sum()) if "stake_final" in bets else 0.0,
        }
        if "actual_result" in bets:
            row["wins"] = int(bets["actual_result"].eq("WIN").sum())
            row["losses"] = int(bets["actual_result"].eq("LOSS").sum())
            row["pushes"] = int(bets["actual_result"].eq("PUSH").sum())
            row["profit_units"] = float(bets["profit_units"].sum()) if "profit_units" in bets else 0.0
            row["win_rate"] = float(row["wins"] / (row["wins"] + row["losses"])) if row["wins"] + row["losses"] else 0.0
            row["roi"] = float(row["profit_units"] / row["stake_units"]) if row["stake_units"] else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _ordered(out: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in COMMON_COLUMNS if col in out.columns]
    return out[cols + [col for col in out.columns if col not in cols]].sort_values(
        ["game_date", "market", "execution_action", "selection_score"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def run_edge_filters(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    winner = filter_winner_edges(predictions)
    ou = filter_ou_edges(predictions)
    combined = pd.concat([winner, ou], ignore_index=True, sort=False)
    orders = combined[combined["execution_action"].eq("BET")].copy().reset_index(drop=True)
    return {
        "orders": orders,
        "audit_candidates": combined,
        "winner_audit": winner,
        "ou_audit": ou,
        "daily_summary": summarize_daily(combined),
        "weekly_summary": summarize_weekly(combined),
        "overall_summary": summarize_overall(combined),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create daily MLB winner and O/U edge-filtered picks.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--winner-daily-cap", default=WinnerEdgeConfig.daily_cap, type=int)
    parser.add_argument("--winner-min-margin", default=WinnerEdgeConfig.min_abs_margin, type=float)
    parser.add_argument("--winner-min-probability", default=WinnerEdgeConfig.min_model_probability, type=float)
    parser.add_argument("--ou-daily-cap", default=OUEdgeConfig.daily_cap, type=int)
    parser.add_argument("--ou-min-total-edge", default=OUEdgeConfig.min_abs_total_edge, type=float)
    parser.add_argument("--ou-min-probability", default=OUEdgeConfig.min_model_probability, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    winner_config = WinnerEdgeConfig(
        daily_cap=args.winner_daily_cap,
        min_abs_margin=args.winner_min_margin,
        min_model_probability=args.winner_min_probability,
    )
    ou_config = OUEdgeConfig(
        daily_cap=args.ou_daily_cap,
        min_abs_total_edge=args.ou_min_total_edge,
        min_model_probability=args.ou_min_probability,
    )
    winner = filter_winner_edges(predictions, config=winner_config)
    ou = filter_ou_edges(predictions, config=ou_config)
    combined = pd.concat([winner, ou], ignore_index=True, sort=False)
    result = {
        "orders": combined[combined["execution_action"].eq("BET")].copy().reset_index(drop=True),
        "audit_candidates": combined,
        "winner_audit": winner,
        "ou_audit": ou,
        "daily_summary": summarize_daily(combined),
        "weekly_summary": summarize_weekly(combined),
        "overall_summary": summarize_overall(combined),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "generated_at_utc": utc_now_iso(),
            "predictions_hash": canonical_hash(predictions),
            "outputs": {name: canonical_hash(frame) for name, frame in result.items()},
            "winner_config": asdict(winner_config),
            "ou_config": asdict(ou_config),
        },
    )
    print(result["overall_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
