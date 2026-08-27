from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import MLBModelConfig
from .governance import canonical_hash, utc_now_iso, validate_games, write_manifest

ROLLING_WINDOWS = MLBModelConfig().rolling_windows
TEAM_FORM_WINDOWS = (7, 14, 30)
MAX_TEAM_HISTORY = max(max(ROLLING_WINDOWS), max(TEAM_FORM_WINDOWS))


@dataclass
class TeamHistory:
    scored: deque[int] = field(default_factory=lambda: deque(maxlen=MAX_TEAM_HISTORY))
    allowed: deque[int] = field(default_factory=lambda: deque(maxlen=MAX_TEAM_HISTORY))
    game_dates: deque[datetime] = field(default_factory=lambda: deque(maxlen=MAX_TEAM_HISTORY))
    venues: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    last_game_date: datetime | None = None


@dataclass
class PitcherHistory:
    team_runs_allowed: deque[int] = field(default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS)))
    team_run_diff: deque[int] = field(default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS)))
    team_total_runs: deque[int] = field(default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS)))
    team_wins: deque[int] = field(default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS)))
    start_dates: deque[datetime] = field(default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS)))
    last_start_date: datetime | None = None
    starts: int = 0


@dataclass
class ParkHistory:
    totals: deque[int] = field(default_factory=lambda: deque(maxlen=100))
    home_scores: deque[int] = field(default_factory=lambda: deque(maxlen=100))


def _mean_last(values: deque[int], window: int) -> float:
    sample = list(values)[-window:]
    if not sample:
        return 0.0
    return float(sum(sample) / len(sample))


def _sum_since(values: deque[int], dates: deque[datetime], game_date: datetime, days: int) -> float:
    total = 0.0
    for value, previous in zip(values, dates):
        delta = (game_date - previous).days
        if 0 < delta <= days:
            total += float(value)
    return total


def _decay_weighted_since(values: deque[int], dates: deque[datetime], game_date: datetime, days: int) -> float:
    total = 0.0
    for value, previous in zip(values, dates):
        delta = (game_date - previous).days
        if 0 < delta <= days:
            total += float(value) / max(delta, 1)
    return total


def _rest_days(last_game_date: datetime | None, game_date: datetime) -> int:
    if last_game_date is None:
        return 7
    return max((game_date - last_game_date).days - 1, 0)


def _games_since(history: TeamHistory, game_date: datetime, days: int) -> int:
    return sum(1 for previous in history.game_dates if 0 < (game_date - previous).days <= days)


def _venue_change(history: TeamHistory, venue_id: int | None) -> int:
    if venue_id is None or not history.venues:
        return 0
    return int(history.venues[-1] != venue_id)


def _starter_rest_days(last_start_date: datetime | None, game_date: datetime) -> int:
    if last_start_date is None:
        return 7
    return max((game_date - last_start_date).days - 1, 0)


def _pitcher_features(prefix: str, row: dict[str, object], history: PitcherHistory, game_date: datetime) -> None:
    row[f"{prefix}_pitcher_starts"] = history.starts
    row[f"{prefix}_starter_rest_days"] = _starter_rest_days(history.last_start_date, game_date)
    row[f"{prefix}_starter_team_ra_l3"] = _mean_last(history.team_runs_allowed, 3)
    row[f"{prefix}_starter_team_run_diff_l3"] = _mean_last(history.team_run_diff, 3)
    row[f"{prefix}_starter_game_total_l3"] = _mean_last(history.team_total_runs, 3)
    row[f"{prefix}_starter_team_win_rate_l3"] = _mean_last(history.team_wins, 3)
    row[f"{prefix}_starter_recent_form_index"] = round(
        _mean_last(history.team_run_diff, 3) - 0.35 * (_mean_last(history.team_runs_allowed, 3) - 4.5),
        3,
    )
    for window in ROLLING_WINDOWS:
        row[f"{prefix}_pitcher_team_ra_l{window}"] = _mean_last(history.team_runs_allowed, window)
        row[f"{prefix}_pitcher_team_run_diff_l{window}"] = _mean_last(history.team_run_diff, window)
        row[f"{prefix}_starter_game_total_l{window}"] = _mean_last(history.team_total_runs, window)
        row[f"{prefix}_starter_team_win_rate_l{window}"] = _mean_last(history.team_wins, window)


def _park_features(row: dict[str, object], history: ParkHistory) -> None:
    row["park_games_tracked"] = len(history.totals)
    row["park_total_runs_l50"] = _mean_last(history.totals, 50)
    row["park_home_score_l50"] = _mean_last(history.home_scores, 50)


def _bullpen_proxy_features(prefix: str, row: dict[str, object], history: TeamHistory, game_date: datetime) -> None:
    for days in [1, 3, 5, 7]:
        games = _games_since(history, game_date, days)
        runs_allowed = _sum_since(history.allowed, history.game_dates, game_date, days)
        runs_scored = _sum_since(history.scored, history.game_dates, game_date, days)
        stress = games * 2.0 + runs_allowed * 0.35
        weighted_stress = _decay_weighted_since(history.allowed, history.game_dates, game_date, days)
        row[f"{prefix}_games_last_{days}_days"] = games
        row[f"{prefix}_bullpen_runs_allowed_proxy_l{days}d"] = runs_allowed
        row[f"{prefix}_bullpen_runs_scored_support_l{days}d"] = runs_scored
        row[f"{prefix}_bullpen_stress_proxy_l{days}d"] = stress
        row[f"{prefix}_bullpen_decay_stress_proxy_l{days}d"] = weighted_stress
        row[f"{prefix}_bullpen_ip_proxy_l{days}d"] = round(games * 3.0, 3)
        row[f"{prefix}_bullpen_fatigue_rate_l{days}d"] = round(stress / max(games, 1), 3)


def _team_form_features(prefix: str, row: dict[str, object], history: TeamHistory) -> None:
    for window in TEAM_FORM_WINDOWS:
        scored = _mean_last(history.scored, window)
        allowed = _mean_last(history.allowed, window)
        total = scored + allowed
        diff = scored - allowed
        row[f"{prefix}_runs_scored_l{window}"] = scored
        row[f"{prefix}_runs_allowed_l{window}"] = allowed
        row[f"{prefix}_run_diff_l{window}"] = diff
        row[f"{prefix}_game_total_l{window}"] = total
        row[f"{prefix}_offense_index_l{window}"] = round(_clip(100.0 * scored / 4.5, 55.0, 145.0), 3) if scored > 0 else 100.0
        row[f"{prefix}_run_prevention_index_l{window}"] = round(_clip(100.0 * allowed / 4.5, 55.0, 145.0), 3) if allowed > 0 else 100.0
    row[f"{prefix}_offense_momentum_7v30"] = round(
        float(row[f"{prefix}_runs_scored_l7"]) - float(row[f"{prefix}_runs_scored_l30"]),
        3,
    )
    row[f"{prefix}_run_prevention_momentum_7v30"] = round(
        float(row[f"{prefix}_runs_allowed_l7"]) - float(row[f"{prefix}_runs_allowed_l30"]),
        3,
    )


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _starter_xfip_proxy(history: PitcherHistory) -> float:
    if history.starts <= 0:
        return 4.5
    run_prevention = _mean_last(history.team_runs_allowed, 20) or _mean_last(history.team_runs_allowed, 10) or 4.5
    run_diff = _mean_last(history.team_run_diff, 20)
    return round(_clip(run_prevention - 0.08 * run_diff, 2.2, 7.2), 3)


def _starter_kbb_proxy(history: PitcherHistory) -> float:
    if history.starts <= 0:
        return 14.0
    run_prevention = _mean_last(history.team_runs_allowed, 20) or 4.5
    run_diff = _mean_last(history.team_run_diff, 20)
    return round(_clip(14.0 + 1.35 * run_diff - 1.8 * (run_prevention - 4.5), 3.0, 30.0), 3)


def _wrc_plus_proxy(history: TeamHistory) -> float:
    recent = 0.50 * _mean_last(history.scored, 30) + 0.30 * _mean_last(history.scored, 14) + 0.20 * _mean_last(history.scored, 7)
    if recent <= 0:
        return 100.0
    return round(_clip(100.0 * recent / 4.5, 55.0, 145.0), 3)


def _barrel_rate_proxy(team_history: TeamHistory, park_history: ParkHistory) -> float:
    scoring = 0.70 * _mean_last(team_history.scored, 20) + 0.30 * _mean_last(team_history.scored, 10)
    park_total = _mean_last(park_history.totals, 50)
    scoring_lift = scoring - 4.5 if scoring > 0 else 0.0
    park_lift = (park_total - 9.0) / 2.0 if park_total > 0 else 0.0
    return round(_clip(7.5 + 0.75 * scoring_lift + 0.35 * park_lift, 3.5, 14.0), 3)


def _bullpen_xfip_proxy(history: TeamHistory) -> float:
    games = _mean_last(history.allowed, 5)
    recent = games if games > 0 else _mean_last(history.allowed, 20)
    if recent <= 0:
        return 4.5
    return round(_clip(recent, 2.8, 7.5), 3)


def _advanced_v2_features(
    row: dict[str, object],
    *,
    home_history: TeamHistory,
    away_history: TeamHistory,
    home_pitcher_history: PitcherHistory,
    away_pitcher_history: PitcherHistory,
    park_history: ParkHistory,
) -> None:
    home_bullpen_xfip = _bullpen_xfip_proxy(home_history)
    away_bullpen_xfip = _bullpen_xfip_proxy(away_history)
    home_barrel = _barrel_rate_proxy(home_history, park_history)
    away_barrel = _barrel_rate_proxy(away_history, park_history)
    park_total = _mean_last(park_history.totals, 50)
    temperature = row.get("temperature_open_meteo_f", row.get("temperature_f"))
    wind = row.get("wind_speed_10m_mph", row.get("wind_mph"))
    try:
        temperature_f = float(temperature)
    except (TypeError, ValueError):
        temperature_f = 70.0
    try:
        wind_mph = float(wind)
    except (TypeError, ValueError):
        wind_mph = 0.0

    row["home_sp_xFIP"] = _starter_xfip_proxy(home_pitcher_history)
    row["away_sp_xFIP"] = _starter_xfip_proxy(away_pitcher_history)
    row["home_sp_kbb"] = _starter_kbb_proxy(home_pitcher_history)
    row["away_sp_kbb"] = _starter_kbb_proxy(away_pitcher_history)
    row["home_wRC_plus_vs_hand"] = _wrc_plus_proxy(home_history)
    row["away_wRC_plus_vs_hand"] = _wrc_plus_proxy(away_history)
    row["combined_barrel_rate"] = round((home_barrel + away_barrel) / 2.0, 3)
    row["bullpen_xFIP_diff"] = round(home_bullpen_xfip - away_bullpen_xfip, 3)
    row["bullpen_fatigue_index"] = round(
        (float(row.get("home_bullpen_stress_proxy_l3d", 0.0)) + float(row.get("away_bullpen_stress_proxy_l3d", 0.0))) / 16.0,
        3,
    )
    row["bullpen_fatigue_diff"] = round(
        float(row.get("home_bullpen_fatigue_rate_l3d", 0.0)) - float(row.get("away_bullpen_fatigue_rate_l3d", 0.0)),
        3,
    )
    row["starter_xFIP_diff"] = round(row["home_sp_xFIP"] - row["away_sp_xFIP"], 3)
    row["starter_kbb_diff"] = round(row["home_sp_kbb"] - row["away_sp_kbb"], 3)
    row["offense_wRC_plus_diff"] = round(row["home_wRC_plus_vs_hand"] - row["away_wRC_plus_vs_hand"], 3)
    row["team_offense_form_diff_l7"] = round(
        float(row.get("home_offense_index_l7", 100.0)) - float(row.get("away_offense_index_l7", 100.0)),
        3,
    )
    row["team_offense_form_diff_l14"] = round(
        float(row.get("home_offense_index_l14", 100.0)) - float(row.get("away_offense_index_l14", 100.0)),
        3,
    )
    row["team_offense_form_diff_l30"] = round(
        float(row.get("home_offense_index_l30", 100.0)) - float(row.get("away_offense_index_l30", 100.0)),
        3,
    )
    row["starter_recent_form_diff"] = round(
        float(row.get("home_starter_recent_form_index", 0.0)) - float(row.get("away_starter_recent_form_index", 0.0)),
        3,
    )
    row["park_run_factor"] = round(_clip((park_total / 9.0) if park_total > 0 else 1.0, 0.75, 1.35), 3)
    open_meteo_index = row.get("weather_run_index_open_meteo")
    try:
        weather_index = float(open_meteo_index)
    except (TypeError, ValueError):
        weather_index = (temperature_f - 70.0) * 0.02 + wind_mph * 0.04
    row["weather_run_index"] = round(_clip(weather_index, -1.0, 1.75), 3)
    row["home_sp_xFIP_source"] = "proxy_pitcher_team_runs_allowed"
    row["away_sp_xFIP_source"] = "proxy_pitcher_team_runs_allowed"
    row["home_sp_kbb_source"] = "proxy_pitcher_run_prevention_and_run_diff"
    row["away_sp_kbb_source"] = "proxy_pitcher_run_prevention_and_run_diff"
    row["wRC_plus_vs_hand_source"] = "proxy_team_runs_scored_no_handedness_source_available"
    row["combined_barrel_rate_source"] = "proxy_scoring_and_park_not_statcast"
    row["bullpen_xFIP_source"] = "proxy_recent_team_runs_allowed"
    row["bullpen_fatigue_source"] = "proxy_recent_games_and_runs_allowed"
    row["park_run_factor_source"] = "proxy_park_total_runs_l50"
    weather_source = row.get("weather_source")
    row["weather_run_index_source"] = (
        "open_meteo_archive"
        if pd.notna(row.get("weather_run_index_open_meteo"))
        else ("espn_weather" if pd.notna(weather_source) and str(weather_source) == "espn_scoreboard" else "neutral_placeholder_until_weather_join")
    )


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    games = validate_games(games)
    games = games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    histories: dict[int, TeamHistory] = defaultdict(TeamHistory)
    pitcher_histories: dict[str, PitcherHistory] = defaultdict(PitcherHistory)
    park_histories: dict[int, ParkHistory] = defaultdict(ParkHistory)
    rows: list[dict[str, object]] = []

    for _, game in games.iterrows():
        game_date = datetime.fromisoformat(str(game["game_date"]))
        home_id = int(game["home_team_id"])
        away_id = int(game["away_team_id"])
        venue_id = None if pd.isna(game.get("venue_id")) else int(game["venue_id"])
        home_history = histories[home_id]
        away_history = histories[away_id]
        home_pitcher = str(game.get("home_probable_pitcher") or "UNKNOWN_HOME")
        away_pitcher = str(game.get("away_probable_pitcher") or "UNKNOWN_AWAY")
        home_pitcher_history = pitcher_histories[f"{home_id}:{home_pitcher}"]
        away_pitcher_history = pitcher_histories[f"{away_id}:{away_pitcher}"]
        park_history = park_histories[venue_id or -1]

        row: dict[str, object] = {
            "game_pk": game["game_pk"],
            "game_date": game["game_date"],
            "season": game["season"],
            "game_type": game.get("game_type"),
            "venue_id": game["venue_id"],
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_probable_pitcher": home_pitcher,
            "away_probable_pitcher": away_pitcher,
            "home_rest_days": _rest_days(home_history.last_game_date, game_date),
            "away_rest_days": _rest_days(away_history.last_game_date, game_date),
            "home_games_last_3_days": _games_since(home_history, game_date, 3),
            "away_games_last_3_days": _games_since(away_history, game_date, 3),
            "home_travel_flag": _venue_change(home_history, venue_id),
            "away_travel_flag": _venue_change(away_history, venue_id),
            "home_score": game["home_score"],
            "away_score": game["away_score"],
            "total_runs": game["total_runs"],
            "home_run_diff": game["home_run_diff"],
            "home_injury_count": game.get("home_injury_injury_count", game.get("home_injury_count", 0.0)),
            "away_injury_count": game.get("away_injury_injury_count", game.get("away_injury_count", 0.0)),
            "home_sp_injury_count": game.get("home_injury_sp_injury_count", game.get("home_sp_injury_count", 0.0)),
            "away_sp_injury_count": game.get("away_injury_sp_injury_count", game.get("away_sp_injury_count", 0.0)),
            "home_rp_injury_count": game.get("home_injury_rp_injury_count", game.get("home_rp_injury_count", 0.0)),
            "away_rp_injury_count": game.get("away_injury_rp_injury_count", game.get("away_rp_injury_count", 0.0)),
            "home_bat_injury_count": game.get("home_injury_bat_injury_count", game.get("home_bat_injury_count", 0.0)),
            "away_bat_injury_count": game.get("away_injury_bat_injury_count", game.get("away_bat_injury_count", 0.0)),
            "home_injury_severity_sum": game.get("home_injury_injury_severity_sum", game.get("home_injury_severity_sum", 0.0)),
            "away_injury_severity_sum": game.get("away_injury_injury_severity_sum", game.get("away_injury_severity_sum", 0.0)),
            "injury_count_diff": game.get("injury_count_diff", 0.0),
            "sp_injury_count_diff": game.get("sp_injury_count_diff", 0.0),
            "rp_injury_count_diff": game.get("rp_injury_count_diff", 0.0),
            "bat_injury_count_diff": game.get("bat_injury_count_diff", 0.0),
            "injury_severity_diff": game.get("injury_severity_diff", 0.0),
        }
        _pitcher_features("home", row, home_pitcher_history, game_date)
        _pitcher_features("away", row, away_pitcher_history, game_date)
        _park_features(row, park_history)
        _bullpen_proxy_features("home", row, home_history, game_date)
        _bullpen_proxy_features("away", row, away_history, game_date)
        _team_form_features("home", row, home_history)
        _team_form_features("away", row, away_history)

        for window in ROLLING_WINDOWS:
            home_rs = _mean_last(home_history.scored, window)
            home_ra = _mean_last(home_history.allowed, window)
            away_rs = _mean_last(away_history.scored, window)
            away_ra = _mean_last(away_history.allowed, window)
            row[f"home_runs_scored_l{window}"] = home_rs
            row[f"home_runs_allowed_l{window}"] = home_ra
            row[f"home_run_diff_l{window}"] = home_rs - home_ra
            row[f"away_runs_scored_l{window}"] = away_rs
            row[f"away_runs_allowed_l{window}"] = away_ra
            row[f"away_run_diff_l{window}"] = away_rs - away_ra

        row["temperature_f"] = game.get("temperature_f", pd.NA)
        row["wind_mph"] = game.get("wind_mph", pd.NA)
        row["humidity_pct"] = game.get("humidity_pct", pd.NA)
        row["weather_condition"] = game.get("weather_condition", pd.NA)
        row["wind_direction"] = game.get("wind_direction", pd.NA)
        row["weather_source"] = game.get("weather_source", pd.NA)
        row["weather_join_matched"] = game.get("weather_join_matched", pd.NA)
        row["temperature_open_meteo_f"] = game.get("temperature_open_meteo_f", pd.NA)
        row["humidity_open_meteo_pct"] = game.get("humidity_open_meteo_pct", pd.NA)
        row["wind_speed_10m_mph"] = game.get("wind_speed_10m_mph", pd.NA)
        row["wind_direction_10m_deg"] = game.get("wind_direction_10m_deg", pd.NA)
        row["wind_direction_bucket"] = game.get("wind_direction_bucket", pd.NA)
        row["precipitation_open_meteo_in"] = game.get("precipitation_open_meteo_in", pd.NA)
        row["dome_or_roof_flag"] = game.get("dome_or_roof_flag", pd.NA)
        row["weather_run_index_open_meteo"] = game.get("weather_run_index_open_meteo", pd.NA)
        row["open_meteo_join_matched"] = game.get("open_meteo_join_matched", pd.NA)
        _advanced_v2_features(
            row,
            home_history=home_history,
            away_history=away_history,
            home_pitcher_history=home_pitcher_history,
            away_pitcher_history=away_pitcher_history,
            park_history=park_history,
        )

        rows.append(row)

        home_score = int(game["home_score"])
        away_score = int(game["away_score"])
        home_history.scored.append(home_score)
        home_history.allowed.append(away_score)
        home_history.game_dates.append(game_date)
        if venue_id is not None:
            home_history.venues.append(venue_id)
        home_history.last_game_date = game_date
        away_history.scored.append(away_score)
        away_history.allowed.append(home_score)
        away_history.game_dates.append(game_date)
        if venue_id is not None:
            away_history.venues.append(venue_id)
        away_history.last_game_date = game_date
        home_pitcher_history.team_runs_allowed.append(away_score)
        home_pitcher_history.team_run_diff.append(home_score - away_score)
        home_pitcher_history.team_total_runs.append(home_score + away_score)
        home_pitcher_history.team_wins.append(int(home_score > away_score))
        home_pitcher_history.start_dates.append(game_date)
        home_pitcher_history.last_start_date = game_date
        home_pitcher_history.starts += 1
        away_pitcher_history.team_runs_allowed.append(home_score)
        away_pitcher_history.team_run_diff.append(away_score - home_score)
        away_pitcher_history.team_total_runs.append(home_score + away_score)
        away_pitcher_history.team_wins.append(int(away_score > home_score))
        away_pitcher_history.start_dates.append(game_date)
        away_pitcher_history.last_start_date = game_date
        away_pitcher_history.starts += 1
        park_history.totals.append(home_score + away_score)
        park_history.home_scores.append(home_score)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pregame MLB features.")
    parser.add_argument("--games", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    games = pd.read_csv(args.games)
    features = build_features(games)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False)
    write_manifest(
        args.out.with_suffix(".manifest.json"),
        {
            "generated_at_utc": utc_now_iso(),
            "input_hash": canonical_hash(games),
            "output_hash": canonical_hash(features),
            "rows": len(features),
            "rolling_windows": ROLLING_WINDOWS,
            "feature_columns": list(features.columns),
        },
    )
    print(f"Wrote {len(features)} feature rows to {args.out}")


if __name__ == "__main__":
    main()
