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

## Quick run

1. Download or export the current candidate files into `inputs/`:

   - Required: `inputs/ml_scored_candidates.csv`
   - Optional totals: `inputs/ou_predictions.csv`
   - Optional totals features: `inputs/features.csv`

   The moneyline file must contain the historical training rows **and** the upcoming slate rows. The optional totals files follow the same convention. Input CSVs are ignored by Git so private/local data will not be committed accidentally.

2. Double-click **`START_APP.bat`** on Windows. (`RUN_PREDICTIONS.bat` now opens the same app.)

The first run creates a local Python environment and installs the dependencies. Your browser opens a private local page at `http://127.0.0.1:8765` with input fields, a **Run predictions** button, status messages, and an edge-picks table. Keep the launcher window open while using the page. The runner automatically selects the earliest game date on or after today. It also saves:

- `output/EDGE_PICKS.md` — easy-to-read edge card
- `output/edge_picks.csv` — spreadsheet-ready picks
- `output/ml_orders.csv` and `output/ou_orders.csv` — full model orders

To request a specific date from PowerShell:

```powershell
.\RUN_PREDICTIONS.bat --date 2026-08-28
```

Important: v2.9 does not create its own current candidate panel from an odds API. It selects edges from the supplied scored candidates and predictions. If no selections pass its filters, the correct output is “No edge picks.” Always verify sportsbook lines are current.
