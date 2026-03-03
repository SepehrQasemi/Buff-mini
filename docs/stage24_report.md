# Stage-24 Report

## What Stage-24 Adds
- Dual sizing modes: `risk_pct` (primary) and `alloc_pct` (secondary).
- Cost-aware notional sizing: `notional = equity * risk_pct / (stop_distance_pct + cost_rt_pct)`.
- Capital-level diagnostics for min-notional/cap bottlenecks.

## Reproducibility
- seed: `42`
- same_seed_all_modes: `True`
- same_data_hash_all_modes: `True`

## Baseline vs Stage-24 Modes
| metric | baseline | risk_pct | alloc_pct |
| --- | ---: | ---: | ---: |
| trade_count | 318.000000 | 239.000000 | 266.000000 |
| zero_trade_pct | 50.000000 | 50.000000 | 50.000000 |
| invalid_order_pct | 0.000000 | 22.061856 | 13.195876 |
| walkforward_executed_true_pct | 50.000000 | 50.000000 | 50.000000 |
| mc_trigger_rate | 50.000000 | 50.000000 | 50.000000 |

## Capital-Level Simulation (risk_pct mode)
| initial_equity | final_equity | return_pct | max_drawdown | trade_count | avg_notional | avg_risk_pct_used | invalid_pct | top_reason |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100.00 | 81.421908 | -18.578092 | 0.315726 | 239.00 | 91.802903 | 0.169233 | 22.061856 | POLICY_CAP_HIT |
| 1000.00 | 710.789079 | -28.921092 | 0.413387 | 318.00 | 920.587341 | 0.074478 | 0.000000 | VALID |
| 10000.00 | 7107.890792 | -28.921092 | 0.413387 | 318.00 | 9184.710228 | 0.023773 | 0.000000 | VALID |
| 100000.00 | 71078.907922 | -28.921092 | 0.413387 | 318.00 | 91711.887663 | 0.020000 | 0.000000 | VALID |

## Why It May Still Choke
- next_bottleneck_gate: `death_execution`
- next_bottleneck_death_rate: `0.683862`

## Top Reject Reasons (risk_pct)
- POLICY_CAP_HIT: 107

## Next Actions
- If `SIZE_TOO_SMALL` dominates small equity tiers: verify minimum tradable size feasibility before alpha changes.
- If cap/margin rejects dominate: recalibrate risk caps rather than weakening validation gates.
- If trade_count remains low despite valid sizing: prioritize signal-density/discovery stages, not sizing.

## Verdict
- `SIZING_ACTIVE_NO_EDGE_CHANGE`
