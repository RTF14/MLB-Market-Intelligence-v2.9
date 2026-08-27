# MLB Market Intelligence v2.9 Results

Generated: 2026-05-02

## What Changed

- Added `src/mlb_model/market_intelligence_v2_9.py`.
- Kept the best ML base from v2.8:
  - OOF-aware meta-models
  - game-result + CLV agreement
  - price/segment gates
  - uncertainty/risk-adjusted ranking
- Kept the best OU idea from v2.8b:
  - 20-feature pruned Over/Under models
  - expected closing-line movement target
  - compact OU game-result model
- Added a two-tier daily edge selector:
  - `STRICT`: original strongest edge gate
  - `BACKFILL`: controlled fallback edge gate
- Goal is 2-4 ML and 2-4 OU picks per day when the slate has enough qualified candidates.
- It does not force bets through segment, game, team, pitcher, or edge gates.

## Backtest Comparison

| Version | Test | Market | Bets | Win Rate | Profit | ROI | CLV Win Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| 2.6 | 2024 | ML | 148 | 51.35% | +5.12 | +3.46% | 100.00% |
| 2.6 | 2024 | OU | 319 | 56.21% | +13.48 | +4.23% | 73.84% |
| 2.6 | 2025 | ML | 78 | 50.00% | +5.03 | +6.45% | 100.00% |
| 2.6 | 2025 | OU | 296 | 56.23% | +11.77 | +3.98% | 74.64% |
| 2.6 | 2026 through May 1 | ML | 10 | 50.00% | -0.07 | -0.67% | 90.00% |
| 2.6 | 2026 through May 1 | OU | 68 | 54.55% | +1.16 | +1.71% | 71.43% |
| 2.7 | 2024 | ML | 109 | 50.46% | +4.36 | +4.00% | 100.00% |
| 2.7 | 2024 | OU | 104 | 51.02% | -4.63 | -4.45% | 54.17% |
| 2.7 | 2025 | ML | 78 | 51.28% | +6.94 | +8.90% | 100.00% |
| 2.7 | 2025 | OU | 128 | 57.98% | +9.40 | +7.34% | 73.02% |
| 2.7 | 2026 through May 1 | ML | 9 | 55.56% | +0.93 | +10.37% | 88.89% |
| 2.7 | 2026 through May 1 | OU | 31 | 56.67% | +1.71 | +5.53% | 64.29% |
| 2.8 | 2024 | ML | 128 | 53.91% | +10.83 | +8.46% | 100.00% |
| 2.8 | 2024 | OU | 104 | 51.02% | -4.63 | -4.45% | 54.17% |
| 2.8 | 2025 | ML | 86 | 52.33% | +8.17 | +9.50% | 100.00% |
| 2.8 | 2025 | OU | 128 | 57.98% | +9.40 | +7.34% | 73.02% |
| 2.8 | 2026 through May 1 | ML | 9 | 66.67% | +2.73 | +30.37% | 88.89% |
| 2.8 | 2026 through May 1 | OU | 31 | 56.67% | +1.71 | +5.53% | 64.29% |
| 2.8b | 2024 | ML | 128 | 53.91% | +10.83 | +8.46% | 100.00% |
| 2.8b | 2024 | OU | 19 | 70.59% | +5.14 | +27.04% | 60.00% |
| 2.8b | 2025 | ML | 86 | 52.33% | +8.17 | +9.50% | 100.00% |
| 2.8b | 2025 | OU | 63 | 66.13% | +13.18 | +20.92% | 81.25% |
| 2.8b | 2026 through May 1 | ML | 9 | 66.67% | +2.73 | +30.37% | 88.89% |
| 2.8b | 2026 through May 1 | OU | 19 | 57.89% | +1.43 | +7.51% | 50.00% |
| 2.9 | 2024 | ML | 160 | 53.12% | +9.83 | +6.15% | 100.00% |
| 2.9 | 2024 | OU | 98 | 55.21% | +2.24 | +2.29% | 63.89% |
| 2.9 | 2025 | ML | 126 | 51.59% | +7.76 | +6.16% | 100.00% |
| 2.9 | 2025 | OU | 127 | 57.38% | +7.35 | +5.79% | 71.88% |
| 2.9 | 2026 through May 1 | ML | 15 | 53.33% | +0.64 | +4.25% | 92.86% |
| 2.9 | 2026 through May 1 | OU | 38 | 55.26% | +1.14 | +2.99% | 75.00% |

## 2.9 Tier Counts

| Test | Market | Strict | Backfill | Days With Picks | Daily Pick Range | Avg Picks/Day |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | ML | 128 | 32 | 115 | 1-4 | 1.39 |
| 2024 | OU | 19 | 79 | 80 | 1-3 | 1.23 |
| 2025 | ML | 86 | 40 | 83 | 1-3 | 1.52 |
| 2025 | OU | 63 | 64 | 91 | 1-3 | 1.40 |
| 2026 through May 1 | ML | 9 | 6 | 12 | 1-2 | 1.25 |
| 2026 through May 1 | OU | 19 | 19 | 26 | 1-3 | 1.46 |

## Read

v2.9 is the more practical daily-card version: it increases volume while staying positive across all three windows for both ML and OU.

But it is not the strongest ROI version. v2.8b is better when we want fewer, higher-conviction OU plays. v2.9 trades some edge quality for daily coverage. The clean operating split is:

- Use 2.8b for highest-conviction OU research/paper plays.
- Use 2.9 when the goal is a steadier card with 2-4 possible edges per market per day.
- Keep ML anchored to v2.8; 2.9 backfill helps volume but lowers 2025 and 2026 ML ROI.

## Outputs

- `reports/market_intelligence_v2_9_backtest_2024/`
- `reports/market_intelligence_v2_9_backtest_2025/`
- `reports/market_intelligence_v2_9_2026_through_may1/`
