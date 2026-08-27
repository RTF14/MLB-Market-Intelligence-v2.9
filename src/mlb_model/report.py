from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def daily_report(audit_candidates: pd.DataFrame) -> pd.DataFrame:
    out = audit_candidates.copy()
    sort_cols = [col for col in ["game_date", "execution_action", "selection_score"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[True, True, False][: len(sort_cols)])
    columns = [
        "game_date",
        "game_pk",
        "away_team",
        "home_team",
        "market",
        "sportsbook_side",
        "execution_action",
        "block_reason",
        "total_line",
        "pred_total",
        "pred_edge",
        "heuristic_probability_score",
        "stake_final",
    ]
    return out[[col for col in columns if col in out.columns]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a concise MLB execution report.")
    parser.add_argument("--audit-candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = pd.read_csv(args.audit_candidates)
    report = daily_report(audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
