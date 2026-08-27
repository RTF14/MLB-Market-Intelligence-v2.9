from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mlb_model.market_intelligence_v2_9 import (  # noqa: E402
    MarketIntelligenceV29Config,
    run_v29,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MLB Market Intelligence v2.9 and create an easy-to-read card for the next slate."
    )
    parser.add_argument(
        "--ml",
        type=Path,
        default=ROOT / "inputs" / "ml_scored_candidates.csv",
        help="Historical panel plus the upcoming moneyline candidates.",
    )
    parser.add_argument(
        "--ou",
        type=Path,
        default=ROOT / "inputs" / "ou_predictions.csv",
        help="Optional historical panel plus upcoming totals predictions.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "inputs" / "features.csv",
        help="Optional feature panel used by the totals model.",
    )
    parser.add_argument("--date", help="Slate date in YYYY-MM-DD format. Defaults to the next date in the input.")
    parser.add_argument("--out", type=Path, default=ROOT / "output")
    return parser.parse_args()


def read_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Missing {label}: {path}\n"
            "Put the current input CSV in the inputs folder and run this command again. "
            "See README.md -> Quick run."
        )
    return pd.read_csv(path, low_memory=False)


def choose_slate_date(frames: list[pd.DataFrame], requested: str | None) -> str:
    dates: set[str] = set()
    for frame in frames:
        if "game_date" in frame:
            dates.update(frame["game_date"].dropna().astype(str).str[:10])
    if requested:
        if requested not in dates:
            raise SystemExit(f"No games for {requested}. Available latest date: {max(dates) if dates else 'none'}")
        return requested
    if not dates:
        raise SystemExit("No game_date values were found in the inputs.")
    today = date.today().isoformat()
    future = sorted(value for value in dates if value >= today)
    return future[0] if future else max(dates)


def american(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return ""
    return f"{number:+.0f}"


def first_present(row: pd.Series, names: list[str], default: object = "") -> object:
    for name in names:
        value = row.get(name)
        if pd.notna(value):
            return value
    return default


def build_card(outputs: dict[str, pd.DataFrame], slate_date: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for market, key in [("ML", "ml_orders"), ("OU", "ou_orders")]:
        orders = outputs.get(key, pd.DataFrame()).copy()
        if orders.empty or "game_date" not in orders:
            continue
        orders = orders[orders["game_date"].astype(str).str[:10].eq(slate_date)]
        for _, row in orders.iterrows():
            away = first_present(row, ["away_team"])
            home = first_present(row, ["home_team"])
            if market == "ML":
                pick = first_present(row, ["display_side", "selected_team", "team"])
                if not pick:
                    pick = home if str(row.get("side", "")).upper() == "HOME" else away
                line = american(first_present(row, ["selected_price", "wager_price", "selected_open_price"]))
                probability = first_present(row, ["ml_game_probability", "model_probability"])
                edge = first_present(row, ["probability_edge", "ml_game_ev_open", "ml_game_ev"])
            else:
                side = str(first_present(row, ["side"])).upper()
                total = first_present(row, ["opening_total", "bet_line", "total_line"])
                pick = f"{side} {total}".strip()
                line = american(first_present(row, ["selected_open_price", "open_price", "price"], -110))
                probability = first_present(row, ["ou_game_probability"])
                edge = first_present(row, ["open_edge", "expected_close_delta_for_side", "ou_game_ev"])
            rows.append(
                {
                    "date": slate_date,
                    "game": f"{away} @ {home}",
                    "market": market,
                    "pick": pick,
                    "line": line,
                    "model_probability": probability,
                    "edge": edge,
                    "tier": first_present(row, ["selection_tier"], "EDGE"),
                }
            )
    return pd.DataFrame(rows, columns=["date", "game", "market", "pick", "line", "model_probability", "edge", "tier"])


def write_markdown(card: pd.DataFrame, slate_date: str, path: Path) -> None:
    lines = [f"# MLB v2.9 edge picks — {slate_date}", ""]
    if card.empty:
        lines += ["No bets passed the v2.9 edge filters for this slate.", "", "Passing is a valid model result."]
    else:
        view = card.copy()
        view["model_probability"] = pd.to_numeric(view["model_probability"], errors="coerce").map(
            lambda x: f"{x:.1%}" if pd.notna(x) else ""
        )
        view["edge"] = pd.to_numeric(view["edge"], errors="coerce").map(
            lambda x: f"{x:.3f}" if pd.notna(x) else ""
        )
        lines += [view.to_markdown(index=False), ""]
    lines += ["Research output only; verify line freshness before acting.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ml = read_required(args.ml, "moneyline candidates")
    ou = pd.read_csv(args.ou, low_memory=False) if args.ou.exists() else None
    features = pd.read_csv(args.features, low_memory=False) if args.features.exists() else None
    slate_date = choose_slate_date([frame for frame in [ml, ou] if frame is not None], args.date)
    season = int(slate_date[:4])
    cfg = MarketIntelligenceV29Config(test_season=season, snapshot_mode="live_paper")
    outputs = run_v29(ml_scored_candidates=ml, ou_predictions=ou, features=features, cfg=cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.out / f"{name}.csv", index=False)
    card = build_card(outputs, slate_date)
    card.to_csv(args.out / "edge_picks.csv", index=False)
    write_markdown(card, slate_date, args.out / "EDGE_PICKS.md")

    print(f"\nMLB v2.9 slate: {slate_date}")
    if card.empty:
        print("No edge picks passed the model filters.")
    else:
        print(card.to_string(index=False))
    print(f"\nSaved: {args.out / 'EDGE_PICKS.md'}")


if __name__ == "__main__":
    main()
