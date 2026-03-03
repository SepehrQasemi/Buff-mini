# Stage-25B Edge Program Report

## Scope
- Family-level quality sweeps under fixed validation and costs.
- Research mode isolates signal quality from exchange minima; live mode replays feasibility.

## Run Context
- run_id: `20260303_023156_a3027d37b4f7_stage25B`
- seed: `42`
- mode: `live`
- dry_run: `False`
- symbols: `['BTC/USDT', 'ETH/USDT']`
- timeframes: `['15m', '30m', '1h', '2h', '4h']`
- cost_levels: `['realistic', 'high']`

## Core Metrics
- rows: `80`
- trade_count_total: `130.000000`
- exp_lcb_best: `0.000000`
- exp_lcb_median: `0.000000`
- expectancy_median: `0.000000`
- maxdd_median: `0.000000`
- zero_trade_pct: `75.000000`
- walkforward_executed_true_pct: `0.000000`
- mc_trigger_rate: `5.000000`

## Per-family Snapshot
| family | rows | trade_count_total | exp_lcb_best | exp_lcb_median | zero_trade_pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| combined | 20 | 0.000000 | 0.000000 | 0.000000 | 100.000000 |
| flow | 20 | 130.000000 | -37.708072 | -176.524947 | 0.000000 |
| price | 20 | 0.000000 | 0.000000 | 0.000000 | 100.000000 |
| volatility | 20 | 0.000000 | 0.000000 | 0.000000 | 100.000000 |

## Best Candidates
- combined:
  - BTC/USDT 15m cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0
  - BTC/USDT 15m cost=high exit=atr_trailing exp_lcb=0.000000 expectancy=0.000000 trades=0
  - BTC/USDT 1h cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0
- flow:
  - BTC/USDT 15m cost=high exit=atr_trailing exp_lcb=-37.708072 expectancy=33.157150 trades=13
  - BTC/USDT 30m cost=high exit=atr_trailing exp_lcb=-40.746133 expectancy=6.736193 trades=9
  - BTC/USDT 15m cost=high exit=fixed_atr exp_lcb=-50.336784 expectancy=21.880076 trades=12
- price:
  - BTC/USDT 15m cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0
  - BTC/USDT 15m cost=high exit=atr_trailing exp_lcb=0.000000 expectancy=0.000000 trades=0
  - BTC/USDT 1h cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0
- volatility:
  - BTC/USDT 15m cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0
  - BTC/USDT 15m cost=high exit=atr_trailing exp_lcb=0.000000 expectancy=0.000000 trades=0
  - BTC/USDT 1h cost=high exit=fixed_atr exp_lcb=0.000000 expectancy=0.000000 trades=0

## Status
- `NO_EDGE_IN_LIVE`

## Artifacts
- runs/20260303_023156_a3027d37b4f7_stage25B/stage25B/family_results.csv
- runs/20260303_023156_a3027d37b4f7_stage25B/stage25B/family_results.json
- runs/20260303_023156_a3027d37b4f7_stage25B/stage25B/best_candidates.json
