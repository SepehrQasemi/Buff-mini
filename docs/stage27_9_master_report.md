# Stage-27.9 Master Report

## DATA STATUS
- coverage_gate_status: `OK`
- coverage_years_by_symbol: `{'BTC/USDT': 4.0, 'ETH/USDT': 4.0}`
- used_symbols: `['BTC/USDT', 'ETH/USDT']`
- snapshot: `DATA_FROZEN_v1` / `c734cebc1e80bf15`

## EXECUTION HEALTH
- death_execution_rate: `0.243182`
- zero_trade_pct: `0.000000`
- invalid_pct: `0.000000`
- walkforward_executed_true_pct: `0.000000`
- mc_trigger_rate: `0.000000`

## ROLLING CONTEXT DISCOVERY
- window_counts: `{'3': {'generated': 10, 'evaluated': 10}, '6': {'generated': 0, 'evaluated': 0}}`
- contextual_counts: `{'ROBUST_CONTEXT_EDGE': 0, 'WEAK_CONTEXT_EDGE': 5, 'NOISE': 165}`

| symbol | timeframe | context | rulelet | class | positive_ratio | exp_lcb_median | trades_median |
|---|---|---|---|---|---:|---:|---:|
| ETH/USDT | 15m | VOLUME_SHOCK | MomentumBurst | WEAK_CONTEXT_EDGE | 1.000 | 148.429208 | 2 |
| ETH/USDT | 1h | VOLUME_SHOCK | MomentumBurst | WEAK_CONTEXT_EDGE | 1.000 | 148.429208 | 2 |
| ETH/USDT | 2h | VOLUME_SHOCK | MomentumBurst | WEAK_CONTEXT_EDGE | 1.000 | 148.429208 | 2 |
| ETH/USDT | 30m | VOLUME_SHOCK | MomentumBurst | WEAK_CONTEXT_EDGE | 1.000 | 148.429208 | 2 |
| ETH/USDT | 4h | VOLUME_SHOCK | MomentumBurst | WEAK_CONTEXT_EDGE | 1.000 | 148.429208 | 2 |

## FEASIBILITY ANALYSIS
| timeframe | feasible_pct_median | recommended_risk_floor_median |
|---|---:|---:|
| 15m | 100.000000 | 0.000095 |
| 1h | 100.000000 | 0.000095 |
| 2h | 100.000000 | 0.000095 |
| 30m | 100.000000 | 0.000095 |
| 4h | 100.000000 | 0.000095 |

## EDGE VERDICT
- verdict: `WEAK_EDGE`
- next_bottleneck: `walkforward_and_mc_not_executing`

## Run IDs
- 24: `20260304_115809_5968d15c9201_stage15_9_trace`
- 25_research: `20260304_115815_e829535e1dda_stage25B`
- 25_live: `20260304_115845_a3027d37b4f7_stage25B`
- 26: `20260304_120151_f7d8fa27e646_stage26`
- 27_9_rolling: `20260304_120218_732731ed0966_stage27_9_roll`
- 27_feasibility: `20260304_120221_290f384a0124_stage15_9_trace`
