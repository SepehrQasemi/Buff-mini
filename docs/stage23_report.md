# Stage-23 Unified Choke Repair Report

- seed: `42`
- baseline_run_id: `20260302_034708_d7bd3e6103dd_stage15_9_trace`
- after_run_id: `20260302_034735_d7bd3e6103dd_stage15_9_trace`
- same_seed: `True`
- same_data_hash: `True`

## Baseline vs After
| metric | baseline | after | delta |
| --- | ---: | ---: | ---: |
| zero_trade_pct | 62.500000 | 62.500000 | 0.000000 |
| invalid_pct | 50.000000 | 50.000000 | 0.000000 |
| walkforward_executed_true_pct | 0.000000 | 0.000000 | 0.000000 |
| mc_trigger_rate | 37.500000 | 37.500000 | 0.000000 |
| death_context | 0.500000 | 0.515693 | 0.015693 |
| death_orders | 0.640051 | 0.649373 | 0.009322 |
| death_execution | 0.797423 | 0.765749 | -0.031674 |

## Top 10 Execution Reject Reasons (Baseline vs After)
| reason | baseline_count | after_count |
| --- | ---: | ---: |
| SIZE_ZERO | 0 | 2775 |
| UNKNOWN | 60 | 10 |

## Evidence Paths
- baseline trace: `C:\dev\Buff-mini\runs\20260302_034708_d7bd3e6103dd_stage15_9_trace\trace`
- after trace: `C:\dev\Buff-mini\runs\20260302_034735_d7bd3e6103dd_stage15_9_trace\trace`
- baseline reject breakdown: `C:\dev\Buff-mini\runs\20260302_034708_d7bd3e6103dd_stage15_9_trace\trace\execution_reject_breakdown.json`
- after reject breakdown: `C:\dev\Buff-mini\runs\20260302_034735_d7bd3e6103dd_stage15_9_trace\trace\execution_reject_breakdown.json`

## Biggest Remaining Bottleneck
- gate: `death_execution`
- death_rate: `0.765749`

## Next Actions
- If still choked: address the single largest death-rate gate first before any new signal complexity.
- If WF execution improved: run constrained timeframe sweeps and keep Stage-8/12 validity gates unchanged.
- If MC still low: increase trade density via signal timing quality, not by lowering MC preconditions.
