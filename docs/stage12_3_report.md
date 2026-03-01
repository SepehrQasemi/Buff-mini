# Stage-12.3 Unblocking Report

- run_id: `20260301_051916_7adfeefe759b_stage12`
- status: `Stage-12.3 FAILED`

## Target Metrics
| metric | value | target | pass |
| --- | ---: | ---: | --- |
| zero_trade_pct | 71.969697 | <= 25.000000 | False |
| walkforward_executed_true_pct | 0.000000 | >= 40.000000 | False |
| MC_trigger_rate | 0.000000 | >= 10.000000 | False |
| invalid_pct | 100.000000 | <= 60.000000 | False |

## Before vs After (Stage-12.1 baseline, if available)
| metric | baseline | stage12_3 |
| --- | ---: | ---: |
| zero_trade_pct | 71.969697 | 71.969697 |
| walkforward_executed_true_pct | 0.000000 | 0.000000 |
| mc_trigger_rate | 0.000000 | 0.000000 |
| invalid_pct | 100.000000 | 100.000000 |

## Reject Reason Breakdown
| reason | pct |
| --- | ---: |
| LOW_USABLE_WINDOWS | 28.030303 |
| ZERO_TRADE | 71.969697 |

## Top 10 VALID by robust_score
- no valid combinations

## Why Stage-12.3 failed and what will be changed in Stage-12.4
- Stage-12.3 soft weighting and adaptive usability improved diagnostics coverage but did not satisfy all target metrics. Stage-12.4 will add a deterministic score-based qualification wrapper with bounded search and explicit trade-rate constraints.
