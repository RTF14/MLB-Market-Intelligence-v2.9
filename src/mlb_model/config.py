from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


MODEL_VERSION = "mlb_baseline_team_form_0_1"
EXECUTION_VERSION = "mlb_ou_execution_0_1"


@dataclass(frozen=True)
class MLBModelConfig:
    rolling_windows: tuple[int, ...] = (5, 10, 20)
    test_fraction: float = 0.20
    min_training_rows: int = 200
    model_version: str = MODEL_VERSION
    random_state: int = 42


@dataclass(frozen=True)
class MLBExecutionConfig:
    market: str = "OU"
    daily_cap: int = 6
    target_edge: float = 1.4
    min_abs_edge: float = 0.75
    max_abs_edge: float = 8.0
    min_model_probability: float = 0.525
    stake_base: float = 1.0
    max_total_exposure: float = 6.0
    max_game_exposure: float = 1.0
    bankroll_units: float = 100.0
    max_stake_pct_bankroll: float = 0.01
    global_kill_switch: bool = False
    require_total_line: bool = True
    require_unique_games: bool = True
    execution_version: str = EXECUTION_VERSION


@dataclass(frozen=True)
class WinnerEdgeConfig:
    daily_cap: int = 5
    min_abs_margin: float = 1.25
    max_abs_margin: float = 6.0
    min_model_probability: float = 0.535
    target_margin: float = 2.0
    stake_base: float = 1.0
    bankroll_units: float = 100.0
    max_stake_pct_bankroll: float = 0.01
    execution_version: str = "mlb_winner_edge_0_1"


@dataclass(frozen=True)
class OUEdgeConfig:
    daily_cap: int = 5
    min_abs_total_edge: float = 0.75
    max_abs_total_edge: float = 8.0
    min_model_probability: float = 0.525
    target_total_edge: float = 1.5
    stake_base: float = 1.0
    bankroll_units: float = 100.0
    max_stake_pct_bankroll: float = 0.01
    execution_version: str = "mlb_ou_edge_0_1"


@dataclass(frozen=True)
class MLBPaths:
    root: Path = Path("documents/MLB")
    raw_dir: Path = Path("documents/MLB/data/raw")
    processed_dir: Path = Path("documents/MLB/data/processed")
    model_dir: Path = Path("documents/MLB/models")
    report_dir: Path = Path("documents/MLB/reports")


@dataclass(frozen=True)
class PipelineConfig:
    model: MLBModelConfig = field(default_factory=MLBModelConfig)
    execution: MLBExecutionConfig = field(default_factory=MLBExecutionConfig)
    winner_edge: WinnerEdgeConfig = field(default_factory=WinnerEdgeConfig)
    ou_edge: OUEdgeConfig = field(default_factory=OUEdgeConfig)
    paths: MLBPaths = field(default_factory=MLBPaths)
