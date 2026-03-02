# Stage-23.6 Sizing Integrity Repair Report

- seed: `42`
- baseline_run_id: `20260302_142515_d7bd3e6103dd_stage15_9_trace`
- after_run_id: `20260302_142603_d7bd3e6103dd_stage15_9_trace`
- same_seed: `True`
- same_data_hash: `True`

## Baseline vs After
| metric | baseline | after | delta |
| --- | ---: | ---: | ---: |
| zero_trade_pct | 62.500000 | 62.500000 | 0.000000 |
| invalid_pct | 50.000000 | 50.000000 | 0.000000 |
| walkforward_executed_true_pct | 0.000000 | 0.000000 | 0.000000 |
| mc_trigger_rate | 37.500000 | 37.500000 | 0.000000 |
| death_context | 0.515693 | 0.515693 | 0.000000 |
| death_orders | 0.649373 | 0.649373 | 0.000000 |
| death_execution | 0.765749 | 0.765749 | 0.000000 |

## Sizing Trace Summary (Baseline vs After)
| counter | baseline | after | delta |
| --- | ---: | ---: | ---: |
| zero_size_count | 0.000000 | 0.000000 | 0.000000 |
| rescued_by_ceil_count | 0.000000 | 0.000000 | 0.000000 |
| bumped_to_min_notional_count | 5705.000000 | 5705.000000 | 0.000000 |
| cap_binding_reject_count | 0.000000 | 0.000000 | 0.000000 |

## Top 10 Reject Reasons (Baseline vs After)
| reason | baseline_count | after_count |
| --- | ---: | ---: |
| SIZE_TOO_SMALL | 2775 | 2775 |
| UNKNOWN | 10 | 10 |

## Improvement Criteria
- size_zero_share_baseline: `0.000000`
- size_zero_share_after: `0.000000`
- criterion_size_zero_pass: `True`
- criterion_choke_pass: `False`
- improvement_sufficient: `False`

## Biggest Remaining Bottleneck
- gate: `death_execution`
- death_rate: `0.765749`

## Deep Why (Triggered)
- likely_root_cause: `capital_too_small_or_min_notional_constraints`
- reason_rank: `[{'reason': 'SIZE_TOO_SMALL', 'count': 2775}]`
- raw_size_quantiles: `{'p01': 0.0, 'p05': 0.0, 'p10': 0.0}`
- step_stats: `{'step_median': 0.001, 'step_p95': 0.001, 'min_notional_median': 10.0}`
- by_symbol_side_top3: `[{'reason': 'SIZE_TOO_SMALL', 'breakdown': [{'symbol': 'BTC/USDT', 'side': 'SHORT', 'count': 770}, {'symbol': 'ETH/USDT', 'side': 'SHORT', 'count': 750}, {'symbol': 'ETH/USDT', 'side': 'LONG', 'count': 660}, {'symbol': 'BTC/USDT', 'side': 'LONG', 'count': 595}]}]`

### Next Actions
- If SIZE_TOO_SMALL dominates: revisit capital/min-notional feasibility before strategy changes.
- If POLICY_CAP_HIT dominates: inspect cap calibration rather than relaxing validation gates.
- If upstream scarcity dominates (RAW_SIGNAL_ZERO/CONTEXT_REJECT): tune signal density before execution changes.

## Evidence Paths
- baseline trace: `C:\dev\Buff-mini\runs\20260302_142515_d7bd3e6103dd_stage15_9_trace\trace`
- after trace: `C:\dev\Buff-mini\runs\20260302_142603_d7bd3e6103dd_stage15_9_trace\trace`
- baseline sizing trace: `C:\dev\Buff-mini\runs\20260302_142515_d7bd3e6103dd_stage15_9_trace\trace\sizing_trace.csv`
- after sizing trace: `C:\dev\Buff-mini\runs\20260302_142603_d7bd3e6103dd_stage15_9_trace\trace\sizing_trace.csv`
