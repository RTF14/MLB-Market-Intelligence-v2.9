from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def canonical_hash(obj: Any) -> str:
    if isinstance(obj, pd.DataFrame):
        sort_keys = [
            col
            for col in ["season", "game_date", "game_pk", "market", "side", "execution_action"]
            if col in obj.columns
        ]
        ordered = obj.sort_index(axis=1)
        if sort_keys:
            ordered = ordered.sort_values(sort_keys, kind="mergesort")
        payload = ordered.to_json(orient="records", date_format="iso", double_precision=12)
    else:
        if is_dataclass(obj):
            obj = asdict(obj)
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def validate_games(frame: pd.DataFrame, *, require_unique_games: bool = True) -> pd.DataFrame:
    required = {
        "game_pk",
        "game_date",
        "season",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
        "total_runs",
        "home_run_diff",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MLB game frame missing required columns: {sorted(missing)}")

    out = frame.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="raise").dt.date.astype(str)
    for col in ["game_pk", "season", "home_team_id", "away_team_id", "home_score", "away_score"]:
        out[col] = pd.to_numeric(out[col], errors="raise").astype(int)
    for col in ["total_runs", "home_run_diff"]:
        out[col] = pd.to_numeric(out[col], errors="raise")

    if (out[["home_score", "away_score", "total_runs"]] < 0).any().any():
        raise ValueError("MLB game frame has impossible negative score values")
    if require_unique_games and out.duplicated(["game_pk"]).any():
        examples = out.loc[out.duplicated(["game_pk"], keep=False), ["game_pk", "game_date"]].head(5)
        raise ValueError(f"MLB game frame has duplicate game_pk rows: {examples.to_dict('records')}")
    return out.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"game_pk", "game_date", "home_score", "away_score", "total_runs", "home_run_diff"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MLB feature frame missing required columns: {sorted(missing)}")
    out = frame.copy()
    if out.empty:
        raise ValueError("MLB feature frame is empty")
    if out["game_pk"].duplicated().any():
        raise ValueError("MLB feature frame has duplicate game_pk values")
    numeric_cols = [col for col in out.columns if col not in {"game_date", "home_team", "away_team"}]
    for col in numeric_cols:
        if out[col].dtype == object:
            converted = pd.to_numeric(out[col], errors="ignore")
            out[col] = converted
    return out.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
