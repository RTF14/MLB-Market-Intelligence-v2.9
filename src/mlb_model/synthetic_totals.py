from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import OUEdgeConfig
from .edge_filters import filter_ou_edges, summarize_daily, summarize_overall, summarize_weekly
from .governance import canonical_hash, utc_now_iso, write_manifest


def add_synthetic_total_line(
    predictions: pd.DataFrame,
    *,
    rolling_days: int = 30,
    min_games: int = 50,
) -> pd.DataFrame:
    required = {"game_date", "game_pk", "total_runs", "pred_total"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Synthetic totals need columns: {sorted(missing)}")

    out = predictions.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise")
    out = out.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    dates = out["game_date"]
    totals = pd.to_numeric(out["total_runs"], errors="raise")
    synthetic_lines = []

    for idx, game_date in enumerate(dates):
        start = game_date - pd.Timedelta(days=rolling_days)
        history = out.loc[(dates < game_date) & (dates >= start), "total_runs"]
        if len(history) < min_games:
            history = totals.iloc[:idx]
        synthetic_lines.append(float(history.mean()) if len(history) else float(totals.mean()))

    out["synthetic_total_line"] = pd.Series(synthetic_lines).round(2)
    out["total_line"] = out["synthetic_total_line"]
    out["synthetic_line_source"] = f"rolling_{rolling_days}_day_actual_total_no_peek"
    out["game_date"] = out["game_date"].dt.date.astype(str)
    return out


def run_synthetic_ou_backtest(
    predictions: pd.DataFrame,
    *,
    rolling_days: int = 30,
    min_games: int = 50,
    ou_daily_cap: int = OUEdgeConfig.daily_cap,
) -> dict[str, pd.DataFrame]:
    with_line = add_synthetic_total_line(predictions, rolling_days=rolling_days, min_games=min_games)
    ou = filter_ou_edges(with_line, config=OUEdgeConfig(daily_cap=ou_daily_cap))
    bets = ou[ou["execution_action"].eq("BET")].copy().reset_index(drop=True)
    return {
        "predictions_with_synthetic_line": with_line,
        "ou_audit": ou,
        "orders": bets,
        "daily_summary": summarize_daily(ou),
        "weekly_summary": summarize_weekly(ou),
        "overall_summary": summarize_overall(ou),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest synthetic MLB O/U lines from model totals vs no-peek baseline.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--rolling-days", default=30, type=int)
    parser.add_argument("--min-games", default=50, type=int)
    parser.add_argument("--ou-daily-cap", default=OUEdgeConfig.daily_cap, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    result = run_synthetic_ou_backtest(
        predictions,
        rolling_days=args.rolling_days,
        min_games=args.min_games,
        ou_daily_cap=args.ou_daily_cap,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "generated_at_utc": utc_now_iso(),
            "predictions_hash": canonical_hash(predictions),
            "rolling_days": args.rolling_days,
            "min_games": args.min_games,
            "ou_daily_cap": args.ou_daily_cap,
            "outputs": {name: canonical_hash(frame) for name, frame in result.items()},
        },
    )
    print(result["overall_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
