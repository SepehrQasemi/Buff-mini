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
| trade_count | 32.000000 | 32.000000 | 32.000000 |
| zero_trade_pct | 0.000000 | 0.000000 | 0.000000 |
| invalid_order_pct | 0.000000 | 0.000000 | 0.000000 |
| walkforward_executed_true_pct | 0.000000 | 0.000000 | 0.000000 |
| mc_trigger_rate | 100.000000 | 100.000000 | 100.000000 |

## Capital-Level Simulation (risk_pct mode)
| initial_equity | final_equity | return_pct | max_drawdown | trade_count | avg_notional | avg_risk_pct_used | invalid_pct | top_reason |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100.00 | 72.547722 | -27.452278 | 0.204381 | 32.00 | 99.220133 | 0.190000 | 0.000000 | VALID |
| 1000.00 | 725.477221 | -27.452278 | 0.204381 | 32.00 | 992.197745 | 0.076276 | 0.000000 | VALID |
| 10000.00 | 7254.772213 | -27.452278 | 0.204381 | 32.00 | 9921.969947 | 0.024741 | 0.000000 | VALID |
| 100000.00 | 72547.722127 | -27.452278 | 0.204381 | 32.00 | 99219.700545 | 0.020000 | 0.000000 | VALID |

## Why It May Still Choke
- next_bottleneck_gate: `death_execution`
- next_bottleneck_death_rate: `0.243182`

## Top Reject Reasons (risk_pct)

## Next Actions
- If `SIZE_TOO_SMALL` dominates small equity tiers: verify minimum tradable size feasibility before alpha changes.
- If cap/margin rejects dominate: recalibrate risk caps rather than weakening validation gates.
- If trade_count remains low despite valid sizing: prioritize signal-density/discovery stages, not sizing.

## Verdict
- `SIZING_ACTIVE_NO_EDGE_CHANGE`
