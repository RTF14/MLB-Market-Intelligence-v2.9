from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_dataset import schedule_rows, write_rows
from .governance import canonical_hash, utc_now_iso, write_manifest
from .statsapi_client import StatsApiClient


def build_history(start_year: int, end_year: int, out: Path, cache_dir: Path | None = None) -> None:
    client = StatsApiClient()
    all_rows = []
    payload_hashes: dict[str, str] = {}
    for year in range(start_year, end_year + 1):
        payload = client.schedule(f"{year}-01-01", f"{year}-12-31")
        payload_hashes[str(year)] = canonical_hash(payload)
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"schedule_{year}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        all_rows.extend(schedule_rows(payload))

    deduped = {}
    for row in sorted(all_rows, key=lambda item: (item["game_date"], item["game_pk"])):
        deduped[int(row["game_pk"])] = row
    rows = list(deduped.values())
    write_rows(rows, out)
    write_manifest(
        out.with_suffix(".manifest.json"),
        {
            "generated_at_utc": utc_now_iso(),
            "source": "statsapi.mlb.com/api/v1/schedule",
            "start_year": start_year,
            "end_year": end_year,
            "rows": len(rows),
            "payload_hashes": payload_hashes,
            "output_hash": canonical_hash(rows),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download multiple MLB seasons from Stats API.")
    parser.add_argument("--start-year", required=True, type=int)
    parser.add_argument("--end-year", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_history(args.start_year, args.end_year, args.out, args.cache_dir)
    print(f"Wrote MLB history to {args.out}")


if __name__ == "__main__":
    main()
