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

### Run entirely on GitHub

One-time setup:

1. Open the repository on GitHub and select **Settings → Secrets and variables → Actions**.
2. Click **New repository secret**.
3. Name it exactly `THE_ODDS_API_KEY`, paste the key as its value, and save it.

To generate predictions:

1. Open the repository's **Actions** tab.
2. Select **Run MLB predictions** in the left sidebar.
3. Click **Run workflow**, then confirm with the green **Run workflow** button.
4. Open the completed run to see the edge-picks table under **Summary**.
5. Optionally download the `mlb-v2-9-predictions-...` artifact for all CSV and Markdown outputs.

The secret is passed only to the prediction job and is not written into the repository or uploaded output.

### Run on Windows

1. Set `THE_ODDS_API_KEY` in your Windows environment. The app uses the existing value automatically.

2. Double-click **`START_APP.bat`** on Windows. (`RUN_PREDICTIONS.bat` opens the same app.)

The first run creates a local Python environment and installs the dependencies. Your browser opens a private local page at `http://127.0.0.1:8765`. Click **Run next-game predictions**. The app fetches the next MLB slate, refreshes completed games, fetches current odds, builds features, runs v2.9, and displays the edge-picks table. Keep the launcher window open while using the page. It also saves:

- `output/EDGE_PICKS.md` — easy-to-read edge card
- `output/edge_picks.csv` — spreadsheet-ready picks
- `output/ml_orders.csv` and `output/ou_orders.csv` — full model orders

To request a specific date from PowerShell:

```powershell
.\RUN_PREDICTIONS.bat --date 2026-08-28
```

If no selections pass the filters, the correct output is “No edge picks.” Always verify sportsbook lines are current. The bundled baseline is trained through 2025 and is research software, not financial advice.
