from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .governance import canonical_hash, utc_now_iso, write_manifest
from .moneyline_classifier_v2_2 import american_implied_probability


SnapshotMode = Literal["historical_backtest", "live_paper"]


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _timestamp_series(df: pd.DataFrame, names: list[str], fallback: pd.Series | None = None) -> pd.Series:
    col = _first_existing(df, names)
    if col is None:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]") if fallback is None else pd.to_datetime(fallback, utc=True, errors="coerce")
    values = pd.to_datetime(df[col], utc=True, errors="coerce")
    if fallback is not None:
        values = values.fillna(pd.to_datetime(fallback, utc=True, errors="coerce"))
    return values


def _date_time_fallback(df: pd.DataFrame, hour: int, minute: int) -> pd.Series:
    dates = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)
    return pd.Series(dates + f"T{hour:02d}:{minute:02d}:00Z", index=df.index)


def _ml_clv_result(open_price: pd.Series, close_price: pd.Series) -> pd.Series:
    open_imp = american_implied_probability(open_price)
    close_imp = american_implied_probability(close_price)
    delta = close_imp - open_imp
    return pd.Series(np.select([delta > 0.0005, delta < -0.0005], ["WIN", "LOSS"], default="PUSH"), index=open_price.index)


def _ou_clv_result(df: pd.DataFrame) -> pd.Series:
    side = df["side"].astype(str)
    open_total = _num(df, "opening_total").fillna(_num(df, "total_line_open"))
    close_total = _num(df, "closing_total").fillna(_num(df, "total_line"))
    over_win = side.eq("OVER") & close_total.gt(open_total)
    over_loss = side.eq("OVER") & close_total.lt(open_total)
    under_win = side.eq("UNDER") & close_total.lt(open_total)
    under_loss = side.eq("UNDER") & close_total.gt(open_total)
    return pd.Series(np.select([over_win | under_win, over_loss | under_loss], ["WIN", "LOSS"], default="PUSH"), index=df.index)


def add_clv_snapshot_fields(
    orders: pd.DataFrame,
    *,
    market: Literal["ML", "OU"],
    mode: SnapshotMode = "historical_backtest",
    sportsbook_source: str | None = None,
) -> pd.DataFrame:
    out = orders.copy()
    if out.empty:
        for col in [
            "bet_odds_timestamp",
            "close_odds_timestamp",
            "game_start_time_utc",
            "clv_locked_before_game",
            "clv_price_delta",
            "clv_implied_probability_delta",
            "clv_result",
            "bet_snapshot_source",
            "closing_snapshot_source",
            "snapshot_mode",
        ]:
            out[col] = []
        return out

    bet_fallback = _date_time_fallback(out, 12, 0) if mode == "historical_backtest" else None
    close_fallback = _date_time_fallback(out, 23, 55) if mode == "historical_backtest" else None
    start_fallback = _date_time_fallback(out, 23, 59) if mode == "historical_backtest" else None

    out["bet_odds_timestamp"] = _timestamp_series(
        out,
        ["bet_odds_timestamp", "selected_open_timestamp", "odds_open_timestamp", "market_open_timestamp_utc", "odds_timestamp"],
        fallback=bet_fallback,
    )
    out["close_odds_timestamp"] = _timestamp_series(
        out,
        ["close_odds_timestamp", "selected_close_timestamp", "odds_close_timestamp", "market_close_timestamp_utc"],
        fallback=close_fallback,
    )
    out["game_start_time_utc"] = _timestamp_series(
        out,
        ["game_start_time_utc", "start_time_utc", "commence_time", "game_datetime_utc"],
        fallback=start_fallback,
    )

    has_close_snapshot = out["close_odds_timestamp"].notna()
    out["clv_locked_before_game"] = has_close_snapshot & out["game_start_time_utc"].notna() & (out["close_odds_timestamp"] <= out["game_start_time_utc"])
    out["snapshot_mode"] = mode
    out["bet_snapshot_source"] = sportsbook_source or out.get("sportsbook", pd.Series("UNKNOWN", index=out.index)).astype(str)
    out["closing_snapshot_source"] = np.where(has_close_snapshot, sportsbook_source or out.get("sportsbook", pd.Series("UNKNOWN", index=out.index)).astype(str), "")

    if market == "ML":
        open_price = _num(out, "wager_price").fillna(_num(out, "selected_open_price"))
        close_price = _num(out, "selected_close_price").fillna(_num(out, "selected_price"))
        out["bet_price"] = open_price
        out["close_price"] = close_price
        out["clv_price_delta"] = close_price - open_price
        out["clv_implied_probability_delta"] = american_implied_probability(close_price) - american_implied_probability(open_price)
        calculated = _ml_clv_result(open_price, close_price)
    else:
        open_total = _num(out, "opening_total").fillna(_num(out, "total_line_open"))
        close_total = _num(out, "closing_total").fillna(_num(out, "total_line"))
        direction = np.where(out["side"].astype(str).eq("UNDER"), -1.0, 1.0)
        out["bet_line"] = open_total
        out["close_line"] = close_total
        out["clv_price_delta"] = direction * (close_total - open_total)
        out["clv_implied_probability_delta"] = np.nan
        calculated = _ou_clv_result(out)

    out["clv_result"] = np.where(out["clv_locked_before_game"], calculated, "PENDING")
    return out


def build_snapshot_tables(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_cols = [col for col in ["season", "game_date", "game_pk", "market", "side", "home_team", "away_team"] if col in orders.columns]
    bet_cols = id_cols + [
        "bet_odds_timestamp",
        "bet_snapshot_source",
        "bet_price",
        "bet_line",
        "wager_price",
        "selected_open_price",
        "opening_total",
    ]
    close_cols = id_cols + [
        "close_odds_timestamp",
        "closing_snapshot_source",
        "game_start_time_utc",
        "clv_locked_before_game",
        "close_price",
        "close_line",
        "selected_close_price",
        "closing_total",
        "clv_price_delta",
        "clv_implied_probability_delta",
        "clv_result",
    ]
    return (
        orders[[col for col in bet_cols if col in orders.columns]].copy(),
        orders[[col for col in close_cols if col in orders.columns]].copy(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add live/paper pre-game closing CLV snapshot fields to MLB orders.")
    parser.add_argument("--orders", required=True, type=Path)
    parser.add_argument("--market", choices=["ML", "OU"], required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=["historical_backtest", "live_paper"], default="historical_backtest")
    parser.add_argument("--sportsbook-source")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orders = pd.read_csv(args.orders, low_memory=False)
    out = add_clv_snapshot_fields(orders, market=args.market, mode=args.mode, sportsbook_source=args.sportsbook_source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    write_manifest(
        args.out.with_suffix(".manifest.json"),
        {
            "generated_at_utc": utc_now_iso(),
            "market": args.market,
            "mode": args.mode,
            "input_hash": canonical_hash(orders),
            "output_hash": canonical_hash(out),
        },
    )
    print(f"Wrote {len(out)} snapshot rows to {args.out}")


if __name__ == "__main__":
    main()
