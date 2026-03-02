# Stage-25B Edge Program Report

## Scope
- Family-level quality sweeps under fixed validation and costs.
- Research mode isolates signal quality from exchange minima; live mode replays feasibility.

## Run Context
- run_id: `20260302_213415_e829535e1dda_stage25B`
- seed: `42`
- mode: `research`
- dry_run: `True`
- symbols: `['BTC/USDT', 'ETH/USDT']`
- timeframes: `['15m', '30m', '1h', '2h', '4h']`
- cost_levels: `['realistic', 'high']`

## Core Metrics
- rows: `160`
- trade_count_total: `1835.000000`
- exp_lcb_best: `0.000000`
- exp_lcb_median: `-20.081494`
- expectancy_median: `0.000000`
- maxdd_median: `0.019311`
- zero_trade_pct: `50.000000`
- walkforward_executed_true_pct: `0.000000`
- mc_trigger_rate: `50.000000`

## Per-family Snapshot
| family | rows | trade_count_total | exp_lcb_best | exp_lcb_median | zero_trade_pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| combined | 40 | 0.000000 | 0.000000 | 0.000000 | 100.000000 |
| flow | 40 | 1180.000000 | -44.165807 | -63.835897 | 0.000000 |
| price | 40 | 655.000000 | -40.162988 | -51.095113 | 0.000000 |
| volatility | 40 | 0.000000 | 0.000000 | 0.000000 | 100.000000 |

## Best Candidates
- combined:
  - BTC/USDT 15m cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0
  - BTC/USDT 15m cost=high exit=atr_trailing exp_lcb=0.000000 expectancy=0.000000 trades=0
  - BTC/USDT 1h cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0
- flow:
  - ETH/USDT 15m cost=realistic exit=atr_trailing exp_lcb=-44.165807 expectancy=-22.676240 trades=30
  - ETH/USDT 1h cost=realistic exit=atr_trailing exp_lcb=-44.165807 expectancy=-22.676240 trades=30
  - ETH/USDT 2h cost=realistic exit=atr_trailing exp_lcb=-44.165807 expectancy=-22.676240 trades=30
- price:
  - BTC/USDT 15m cost=high exit=atr_trailing exp_lcb=-40.162988 expectancy=-9.523158 trades=24
  - BTC/USDT 1h cost=high exit=atr_trailing exp_lcb=-40.162988 expectancy=-9.523158 trades=24
  - BTC/USDT 2h cost=high exit=atr_trailing exp_lcb=-40.162988 expectancy=-9.523158 trades=24
- volatility:
  - BTC/USDT 15m cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0
  - BTC/USDT 15m cost=high exit=atr_trailing exp_lcb=0.000000 expectancy=0.000000 trades=0
  - BTC/USDT 1h cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0

## Status
- `NO_EDGE_IN_RESEARCH`

## Artifacts
- runs/20260302_213415_e829535e1dda_stage25B/stage25B/family_results.csv
- runs/20260302_213415_e829535e1dda_stage25B/stage25B/family_results.json
- runs/20260302_213415_e829535e1dda_stage25B/stage25B/best_candidates.json
