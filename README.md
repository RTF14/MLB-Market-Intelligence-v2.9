# MLB Market Intelligence v2.9

MLB moneyline and over/under market-intelligence research code, including the v2.9 daily edge selector and its summarized 2024, 2025, and 2026-through-May-1 backtest outputs.

## Version 2.9

The entry point is `src/mlb_model/market_intelligence_v2_9.py`. Version 2.9 adds daily backfill selection to target 2–4 possible edges per market while retaining the stricter upstream model and governance logic.

The implementation depends on the neighboring modules under `src/mlb_model`, which are included so the package remains internally complete.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

View command-line options with:

```powershell
$env:PYTHONPATH = "src"
python -m mlb_model.market_intelligence_v2_9 --help
```

Running a backtest requires the prediction, odds, and game datasets referenced by the command-line arguments. Those source datasets are not included in this repository.

## Results

The main findings are in `reports/market_intelligence_v2_9_tests/RESULTS.md`. Supporting manifests, model metrics, orders, snapshots, and summary tables are included for:

- 2024 backtest
- 2025 backtest
- 2026 through May 1

Large generated `*_scored_candidates.csv` files and raw source datasets are intentionally excluded from Git history. They can be regenerated from the source datasets.
