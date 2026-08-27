from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .config import OUEdgeConfig, WinnerEdgeConfig
from .edge_filters import (
    filter_ou_edges,
    filter_winner_edges,
)
from .governance import canonical_hash, utc_now_iso, write_manifest
from .synthetic_totals import add_synthetic_total_line


EXECUTION_VERSION = "mlb_execution_v1_0"


@dataclass(frozen=True)
class MLBExecutionV10Config:
    winner_daily_cap: int = 4
    winner_min_margin: float = 1.25
    winner_min_probability: float = 0.535
    winner_enabled: bool = True

    ou_daily_cap: int = 4
    ou_min_total_edge: float = 0.75
    ou_min_probability: float = 0.525
    ou_enabled: bool = True
    ou_mode: Literal["market", "synthetic", "both"] = "both"
    synthetic_rolling_days: int = 30
    synthetic_min_games: int = 50

    stake_base: float = 1.0
    bankroll_units: float = 100.0
    max_stake_pct_bankroll: float = 0.01
    execution_version: str = EXECUTION_VERSION


def _winner_config(cfg: MLBExecutionV10Config) -> WinnerEdgeConfig:
    return WinnerEdgeConfig(
        daily_cap=cfg.winner_daily_cap,
        min_abs_margin=cfg.winner_min_margin,
        min_model_probability=cfg.winner_min_probability,
        stake_base=cfg.stake_base,
        bankroll_units=cfg.bankroll_units,
        max_stake_pct_bankroll=cfg.max_stake_pct_bankroll,
    )


def _ou_config(cfg: MLBExecutionV10Config) -> OUEdgeConfig:
    return OUEdgeConfig(
        daily_cap=cfg.ou_daily_cap,
        min_abs_total_edge=cfg.ou_min_total_edge,
        min_model_probability=cfg.ou_min_probability,
        stake_base=cfg.stake_base,
        bankroll_units=cfg.bankroll_units,
        max_stake_pct_bankroll=cfg.max_stake_pct_bankroll,
    )


def _tag_mode(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = frame.copy()
    out["execution_mode"] = mode
    out["execution_version"] = EXECUTION_VERSION
    return out


def _mode_summary(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    group_cols = ["execution_mode", "market"]
    if period == "daily":
        group_cols.append("game_date")
    elif period == "weekly":
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
        bets = group[group["execution_action"].eq("BET")]
        row.update(
            {
                "games": int(len(group)),
                "picks": int(len(bets)),
                "avg_edge_abs": float(bets["edge_abs"].mean()) if len(bets) else 0.0,
                "avg_probability": float(bets["heuristic_probability_score"].mean()) if len(bets) else 0.0,
                "stake_units": float(bets["stake_final"].sum()) if "stake_final" in bets else 0.0,
                "wins": int(bets["actual_result"].eq("WIN").sum()) if "actual_result" in bets else 0,
                "losses": int(bets["actual_result"].eq("LOSS").sum()) if "actual_result" in bets else 0,
                "pushes": int(bets["actual_result"].eq("PUSH").sum()) if "actual_result" in bets else 0,
                "profit_units": float(bets["profit_units"].sum()) if "profit_units" in bets else 0.0,
            }
        )
        decisions = row["wins"] + row["losses"]
        row["win_rate"] = float(row["wins"] / decisions) if decisions else 0.0
        row["roi"] = float(row["profit_units"] / row["stake_units"]) if row["stake_units"] else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def execute_v1_0(predictions: pd.DataFrame, *, config: MLBExecutionV10Config | None = None) -> dict[str, pd.DataFrame]:
    cfg = config or MLBExecutionV10Config()
    outputs: dict[str, pd.DataFrame] = {}
    audit_parts: list[pd.DataFrame] = []

    if cfg.winner_enabled:
        winner = filter_winner_edges(predictions, config=_winner_config(cfg))
        winner = _tag_mode(winner, "market_independent_winner")
        outputs["winner_audit"] = winner
        audit_parts.append(winner)

    if cfg.ou_enabled and cfg.ou_mode in {"market", "both"}:
        market_ou = filter_ou_edges(predictions, config=_ou_config(cfg))
        market_ou = _tag_mode(market_ou, "market_ou")
        outputs["market_ou_audit"] = market_ou
        audit_parts.append(market_ou)

    if cfg.ou_enabled and cfg.ou_mode in {"synthetic", "both"}:
        synthetic_predictions = add_synthetic_total_line(
            predictions,
            rolling_days=cfg.synthetic_rolling_days,
            min_games=cfg.synthetic_min_games,
        )
        synthetic_ou = filter_ou_edges(synthetic_predictions, config=_ou_config(cfg))
        synthetic_ou = _tag_mode(synthetic_ou, "synthetic_ou")
        outputs["synthetic_ou_audit"] = synthetic_ou
        outputs["predictions_with_synthetic_line"] = synthetic_predictions
        audit_parts.append(synthetic_ou)

    audit = pd.concat(audit_parts, ignore_index=True, sort=False) if audit_parts else pd.DataFrame()
    orders = audit[audit["execution_action"].eq("BET")].copy().reset_index(drop=True) if not audit.empty else pd.DataFrame()
    outputs.update(
        {
            "orders": orders,
            "audit_candidates": audit,
            "daily_summary": _mode_summary(audit, "daily"),
            "weekly_summary": _mode_summary(audit, "weekly"),
            "overall_summary": _mode_summary(audit, "overall"),
        }
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLB execution v1.0.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--winner-daily-cap", default=MLBExecutionV10Config.winner_daily_cap, type=int)
    parser.add_argument("--winner-min-margin", default=MLBExecutionV10Config.winner_min_margin, type=float)
    parser.add_argument("--winner-min-probability", default=MLBExecutionV10Config.winner_min_probability, type=float)
    parser.add_argument("--ou-daily-cap", default=MLBExecutionV10Config.ou_daily_cap, type=int)
    parser.add_argument("--ou-min-total-edge", default=MLBExecutionV10Config.ou_min_total_edge, type=float)
    parser.add_argument("--ou-min-probability", default=MLBExecutionV10Config.ou_min_probability, type=float)
    parser.add_argument("--ou-mode", choices=["market", "synthetic", "both"], default=MLBExecutionV10Config.ou_mode)
    parser.add_argument("--synthetic-rolling-days", default=MLBExecutionV10Config.synthetic_rolling_days, type=int)
    parser.add_argument("--synthetic-min-games", default=MLBExecutionV10Config.synthetic_min_games, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    config = MLBExecutionV10Config(
        winner_daily_cap=args.winner_daily_cap,
        winner_min_margin=args.winner_min_margin,
        winner_min_probability=args.winner_min_probability,
        ou_daily_cap=args.ou_daily_cap,
        ou_min_total_edge=args.ou_min_total_edge,
        ou_min_probability=args.ou_min_probability,
        ou_mode=args.ou_mode,
        synthetic_rolling_days=args.synthetic_rolling_days,
        synthetic_min_games=args.synthetic_min_games,
    )
    outputs = execute_v1_0(predictions, config=config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "execution_version": EXECUTION_VERSION,
            "generated_at_utc": utc_now_iso(),
            "predictions_hash": canonical_hash(predictions),
            "config": asdict(config),
            "outputs": {name: canonical_hash(frame) for name, frame in outputs.items()},
        },
    )
    if "overall_summary" in outputs:
        print(outputs["overall_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
