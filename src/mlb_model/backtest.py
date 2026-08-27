from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .execution import execute_mlb_ou


def grade_ou(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"total_runs", "total_line", "side"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Cannot grade O/U without columns: {sorted(missing)}")
    out = predictions.copy()
    actual_over = out["total_runs"] > out["total_line"]
    actual_under = out["total_runs"] < out["total_line"]
    out["result"] = "PUSH"
    out.loc[out["side"].eq("OVER") & actual_over, "result"] = "WIN"
    out.loc[out["side"].eq("OVER") & actual_under, "result"] = "LOSS"
    out.loc[out["side"].eq("UNDER") & actual_under, "result"] = "WIN"
    out.loc[out["side"].eq("UNDER") & actual_over, "result"] = "LOSS"
    out["profit_units"] = 0.0
    out.loc[out["result"].eq("WIN"), "profit_units"] = out["stake_final"] * 0.9091
    out.loc[out["result"].eq("LOSS"), "profit_units"] = -out["stake_final"]
    return out


def backtest_ou(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    executed = execute_mlb_ou(predictions)
    audit = executed["audit_candidates"]
    graded = grade_ou(audit) if "total_runs" in audit.columns else audit
    bets = graded[graded["execution_action"].eq("BET")].copy()
    summary = pd.DataFrame(
        [
            {
                "bets": int(len(bets)),
                "wins": int(bets["result"].eq("WIN").sum()) if "result" in bets else 0,
                "losses": int(bets["result"].eq("LOSS").sum()) if "result" in bets else 0,
                "pushes": int(bets["result"].eq("PUSH").sum()) if "result" in bets else 0,
                "profit_units": float(bets["profit_units"].sum()) if "profit_units" in bets else 0.0,
                "roi": float(bets["profit_units"].sum() / bets["stake_final"].sum())
                if "profit_units" in bets and bets["stake_final"].sum() > 0
                else 0.0,
            }
        ]
    )
    return {"summary": summary, "graded": graded, "orders": executed["orders"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest MLB O/U predictions with execution gates.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    result = backtest_ou(predictions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
