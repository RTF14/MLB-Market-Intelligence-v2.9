from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .execution import execute_mlb_ou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run governed MLB O/U execution.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    result = execute_mlb_ou(predictions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    print(f"Wrote execution outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
