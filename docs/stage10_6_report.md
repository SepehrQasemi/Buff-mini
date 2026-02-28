# Stage-10.6 Report

## What Changed
- Switched activation decisions to score-only regime usage.
- Clamped activation multipliers to a strict soft band.
- Reduced default exits to fixed ATR and ATR trailing.
- Added sandbox ranking with drag-aware stress penalty.

## Sandbox Selection
- enabled_signals: `['ATR_DistanceRevert', 'BreakoutRetest', 'RangeFade', 'VolCompressionBreakout']`
- disabled_signals: `['BollingerSnapBack', 'MA_SlopePullback']`
- ranking table: `C:\dev\Buff-mini\runs\20260228_122344_7e18484fd058_stage10_sandbox\sandbox_rankings.csv`

## Comparisons (Dry Run)
- pre_run_id: `20260228_111937_2d4caf91e389_stage10`
- stage10_6_run_id: `20260228_122912_9a27f1e5886a_stage10`
- walkforward: `N/A -> N/A`
- usable_windows: `0 -> 0`
- single_label_warning: `False`

| metric | baseline | stage10_pre | stage10_6 |
| --- | ---: | ---: | ---: |
| trade_count | 519.000000 | 450.000000 | 450.000000 |
| profit_factor | 0.618459 | 0.455899 | 0.424441 |
| expectancy | -32.934793 | -25.574440 | -26.813456 |
| max_drawdown | 0.441529 | 0.601970 | 0.643389 |
| pf_adj | 0.651986 | 0.510309 | 0.481997 |
| exp_lcb | -34.028256 | -25.625012 | -26.839207 |

## Comparisons (Real Data)
- available: `True`
- pre_run_id: `20260228_112303_c3a6713a904d_stage10`
- stage10_6_run_id: `20260228_123104_791590f09a7d_stage10`
- walkforward: `UNSTABLE -> UNSTABLE`
- usable_windows: `16 -> 16`
- single_label_warning: `False`

| metric | baseline | stage10_pre | stage10_6 |
| --- | ---: | ---: | ---: |
| trade_count | 6275.000000 | 2923.000000 | 5627.000000 |
| profit_factor | 0.616272 | 0.322475 | 0.265562 |
| expectancy | -7.830390 | -3.042735 | -3.518895 |
| max_drawdown | 0.954500 | 0.970305 | 0.990047 |
| pf_adj | 0.619305 | 0.333869 | 0.272031 |
| exp_lcb | -7.896892 | -4.974870 | -3.520591 |

## Trade Count Guard
- max_drop_pct: `10.00`
- observed_drop_pct: `10.326693`
- pass: `False`
- justified_by_robust_improvement: `False`

## Determinism
- pass: `True`
- notes: `PASS`
- signature_previous: `29e4517119c707f17a764e3c`
- signature_latest: `29e4517119c707f17a764e3c`

## Known Limitations
- Stage-10.6 ranking is currently single-exit-first (fixed_atr) before A/B exit sweeps.
- Drag penalty is sandbox-only and not yet part of the main optimizer objective.
