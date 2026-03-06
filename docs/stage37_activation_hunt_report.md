# Stage-37 Activation Hunt Report

## Run Context
- stage28_run_id: `20260306_152631_2a029423a621_stage28`
- seed: `42`
- mode: `both`
- budget_small: `True`

## Strict vs Hunt
| metric | strict | hunt |
| --- | ---: | ---: |
| raw_signal_count | 87650 | 87650 |
| post_threshold_count | 87650 | 87650 |
| post_cost_gate_count | 42463 | 42463 |
| post_feasibility_count | 42463 | 42463 |
| final_trade_count | 42463.000 | 42463.000 |
| activation_rate | 0.484461 | 0.484461 |

## Gate Killer
- dominant_gate: `cost_gate`
- dropped_count: `45187`

## Threshold Sensitivity
| threshold | selected_rows | post_feasibility_count | avg_context_quality |
| ---: | ---: | ---: | ---: |
| 0.0000 | 42463 | 42463 | 39.757079 |
| 0.0200 | 0 | 0 | 0.000000 |
| 0.0400 | 0 | 0 | 0.000000 |
| 0.0600 | 0 | 0 | 0.000000 |
| 0.0800 | 0 | 0 | 0.000000 |
| 0.1000 | 0 | 0 | 0.000000 |
| 0.1200 | 0 | 0 | 0.000000 |
| 0.1500 | 0 | 0 | 0.000000 |
| 0.2000 | 0 | 0 | 0.000000 |
| 0.2500 | 0 | 0 | 0.000000 |
| 0.3000 | 0 | 0 | 0.000000 |

## Top Reject Reasons (Strict)
- none

## Family Gate Summary (Hunt)
| family | raw | post_threshold | post_cost | post_feasibility | activation_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
