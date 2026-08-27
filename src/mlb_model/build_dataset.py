from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

from .statsapi_client import StatsApiClient
from .governance import canonical_hash, write_manifest, utc_now_iso


FINAL_STATES = {"Final", "Game Over", "Completed Early"}


def _team_value(game: dict[str, Any], side: str, key: str, default: Any = None) -> Any:
    return game.get("teams", {}).get(side, {}).get(key, default)


def _team_id(game: dict[str, Any], side: str) -> int | None:
    team = _team_value(game, side, "team", {})
    return team.get("id")


def _team_name(game: dict[str, Any], side: str) -> str | None:
    team = _team_value(game, side, "team", {})
    return team.get("name")


def _pitcher_name(game: dict[str, Any], side: str) -> str | None:
    pitcher = _team_value(game, side, "probablePitcher", {})
    return pitcher.get("fullName")


def schedule_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day in payload.get("dates", []):
        game_date = day.get("date")
        for game in day.get("games", []):
            status = game.get("status", {}).get("detailedState")
            if status not in FINAL_STATES:
                continue

            home_score = _team_value(game, "home", "score")
            away_score = _team_value(game, "away", "score")
            if home_score is None or away_score is None:
                continue

            rows.append(
                {
                    "game_pk": game.get("gamePk"),
                    "game_date": game_date,
                    "season": game.get("season"),
                    "game_type": game.get("gameType"),
                    "venue_id": game.get("venue", {}).get("id"),
                    "venue_name": game.get("venue", {}).get("name"),
                    "home_team_id": _team_id(game, "home"),
                    "home_team": _team_name(game, "home"),
                    "away_team_id": _team_id(game, "away"),
                    "away_team": _team_name(game, "away"),
                    "home_probable_pitcher": _pitcher_name(game, "home"),
                    "away_probable_pitcher": _pitcher_name(game, "away"),
                    "home_score": home_score,
                    "away_score": away_score,
                    "total_runs": int(home_score) + int(away_score),
                    "home_run_diff": int(home_score) - int(away_score),
                }
            )
    deduped: dict[int, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (item["game_date"], item["game_pk"])):
        deduped[int(row["game_pk"])] = row
    return list(deduped.values())


def write_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "game_pk",
        "game_date",
        "season",
        "game_type",
        "venue_id",
        "venue_name",
        "home_team_id",
        "home_team",
        "away_team_id",
        "away_team",
        "home_probable_pitcher",
        "away_probable_pitcher",
        "home_score",
        "away_score",
        "total_runs",
        "home_run_diff",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download completed MLB games from Stats API.")
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache-json", type=Path, help="Optional path for the raw schedule payload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = StatsApiClient()
    payload = client.schedule(args.start_date, args.end_date)
    if args.cache_json:
        args.cache_json.parent.mkdir(parents=True, exist_ok=True)
        args.cache_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    rows = schedule_rows(payload)
    write_rows(rows, args.out)
    write_manifest(
        args.out.with_suffix(".manifest.json"),
        {
            "generated_at_utc": utc_now_iso(),
            "source": "statsapi.mlb.com/api/v1/schedule",
            "start_date": args.start_date,
            "end_date": args.end_date,
            "rows": len(rows),
            "payload_hash": canonical_hash(payload),
            "output_hash": canonical_hash(rows),
        },
    )
    print(f"Wrote {len(rows)} games to {args.out}")


if __name__ == "__main__":
    main()
