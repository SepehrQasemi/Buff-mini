# Stage-25B Edge Program Report

## Scope
- Family-level quality sweeps under fixed validation and costs.
- Research mode isolates signal quality from exchange minima; live mode replays feasibility.

## Run Context
- run_id: `20260303_045817_127db8e77b93_stage25B`
- seed: `42`
- mode: `live`
- dry_run: `False`
- symbols: `['BTC/USDT', 'ETH/USDT']`
- timeframes: `['4h']`
- cost_levels: `['realistic', 'high']`

## Core Metrics
- rows: `32`
- trade_count_total: `1036.000000`
- exp_lcb_best: `320.054371`
- exp_lcb_median: `0.000000`
- expectancy_median: `0.000000`
- maxdd_median: `0.000000`
- zero_trade_pct: `62.500000`
- walkforward_executed_true_pct: `50.000000`
- mc_trigger_rate: `25.000000`

## Per-family Snapshot
| family | rows | trade_count_total | exp_lcb_best | exp_lcb_median | zero_trade_pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| combined | 8 | 0.000000 | 0.000000 | 0.000000 | 100.000000 |
| flow | 8 | 888.000000 | 320.054371 | -38.594064 | 0.000000 |
| price | 8 | 148.000000 | 0.000000 | -36.290815 | 50.000000 |
| volatility | 8 | 0.000000 | 0.000000 | 0.000000 | 100.000000 |

## Best Candidates
- combined:
  - BTC/USDT 4h cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0
  - BTC/USDT 4h cost=high exit=atr_trailing exp_lcb=0.000000 expectancy=0.000000 trades=0
  - ETH/USDT 4h cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0
- flow:
  - ETH/USDT 4h cost=high exit=fixed_atr exp_lcb=320.054371 expectancy=320.054371 trades=1
  - ETH/USDT 4h cost=realistic exit=fixed_atr exp_lcb=239.866640 expectancy=239.866640 trades=1
  - BTC/USDT 4h cost=high exit=atr_trailing exp_lcb=-29.540531 expectancy=-21.588539 trades=240
- price:
  - ETH/USDT 4h cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0
  - ETH/USDT 4h cost=high exit=atr_trailing exp_lcb=0.000000 expectancy=0.000000 trades=0
  - ETH/USDT 4h cost=realistic exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0
- volatility:
  - BTC/USDT 4h cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0
  - BTC/USDT 4h cost=high exit=atr_trailing exp_lcb=0.000000 expectancy=0.000000 trades=0
  - ETH/USDT 4h cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0

## Status
- `NO_EDGE_IN_LIVE`

## Artifacts
- runs/20260303_045817_127db8e77b93_stage25B/stage25B/family_results.csv
- runs/20260303_045817_127db8e77b93_stage25B/stage25B/family_results.json
- runs/20260303_045817_127db8e77b93_stage25B/stage25B/best_candidates.json
