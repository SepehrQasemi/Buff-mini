# Stage-13/14 Forensic Root-Cause Report

## Executive Summary
- git_head: `353817124f9dd30c103276a8332e2c814a7a4603`
- runtime_seconds: `95.886`
- final_conclusion: `BUG_FOUND_AND_FIXED`
- raw_evidence: `docs/stage13_14_forensic_root_cause_raw.json`

## Baseline (Before)
- `{'master_final_verdict': 'NO_EDGE', 'stage13_combined': {'invalid_pct': 75.0, 'zero_trade_pct': 25.0, 'walkforward_executed_true_pct': 100.0, 'mc_trigger_rate': 75.0, 'best_exp_lcb': 49.78228993550447}, 'stage14_nested': {'folds_evaluated': 16, 'consistency': 0.0625, 'classification': 'NO_EDGE'}}`

## 20 Checks
| # | Check | PASS/FAIL | Root Cause Tag | Evidence Pointer |
| --- | --- | --- | --- | --- |
| 1 | Data Integrity & Size | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[1]` |
| 2 | Resample/Derived Timeframe Correctness | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[2]` |
| 3 | Feature Cache Semantics | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[3]` |
| 4 | Leakage Harness Coverage | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[4]` |
| 5 | Signal Score Range & Distribution | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[5]` |
| 6 | Threshold Application Logic | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[6]` |
| 7 | Entry Construction | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[7]` |
| 8 | Exit Construction & Engine Semantics | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[8]` |
| 9 | Trade Generation Sanity | PASS | TRADE_SANITY | `docs/stage13_14_forensic_root_cause_raw.json#checks[9]` |
| 10 | Zero-trade Causes Attribution | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[10]` |
| 11 | NaN / Masking Safety | PASS | NAN_PROPAGATION | `docs/stage13_14_forensic_root_cause_raw.json#checks[11]` |
| 12 | Walkforward v2 Preconditions | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[12]` |
| 13 | Walkforward Window Slicing Overlap | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[13]` |
| 14 | Monte Carlo Preconditions | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[14]` |
| 15 | Cost Model v2 Finite-Safety and Magnitude | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[15]` |
| 16 | Execution Drag Sensitivity | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[16]` |
| 17 | ML Fold Construction (Stage-14.3) | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[17]` |
| 18 | ML Regularization / Feature Limits | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[18]` |
| 19 | Composer / Ensemble Wiring | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[19]` |
| 20 | Reporting / Metric Integrity | PASS | OK | `docs/stage13_14_forensic_root_cause_raw.json#checks[20]` |

## Impact Drivers (Ranked)
| Rank | Driver | Variant | ?exp_lcb | ?invalid_pct | ?wf_pct | ?tpm |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | family | price | -53.673640 | 25.000000 | 0.000000 | -4.187985 |
| 2 | threshold | price:med | -53.673640 | 25.000000 | 0.000000 | -4.187985 |
| 3 | threshold | price:high | -46.633596 | 25.000000 | 0.000000 | -6.354902 |
| 4 | threshold | flow:high | -23.334238 | -75.000000 | 0.000000 | 10.730409 |
| 5 | family | flow | -20.422243 | -75.000000 | 0.000000 | 17.689547 |
| 6 | threshold | flow:med | -20.422243 | -75.000000 | 0.000000 | 17.689547 |
| 7 | threshold | price:low | -24.326892 | -25.000000 | 0.000000 | 2.229425 |
| 8 | threshold | flow:low | -17.851768 | -75.000000 | 0.000000 | 28.399120 |
| 9 | stage14 | weighting_best | 0.000000 | -75.000000 | -100.000000 | -6.771617 |
| 10 | walkforward | disabled_proxy | 0.000000 | 25.000000 | -100.000000 | 0.000000 |

## Fixes Applied
- BUG_INVALID_PCT_METRIC_FIXED via check `20`

## Re-run Results (After)
- `{'before': {'master_final_verdict': 'NO_EDGE', 'stage13_combined': {'invalid_pct': 75.0, 'zero_trade_pct': 25.0, 'walkforward_executed_true_pct': 100.0, 'mc_trigger_rate': 75.0, 'best_exp_lcb': 49.78228993550447}, 'stage14_nested': {'folds_evaluated': 16, 'consistency': 0.0625, 'classification': 'NO_EDGE'}}, 'after': {'stage13_combined': {'zero_trade_pct': 37.5, 'invalid_pct': 75.0, 'walkforward_executed_true_pct': 100.0, 'mc_trigger_rate': 62.5, 'tpm': 6.7716170853108, 'trade_count': 81.25}, 'stage14_nested': {'stage': '14.3', 'folds_evaluated': 16, 'consistency': 0.0625, 'classification': 'NO_EDGE'}}}`

## Warnings
- none

## Limitations
- none

## What To Do Next
1. Improve signal density without lowering WF/MC gates (score shape + thresholding by regime).
2. Broaden free-data families (cross-symbol/session features) with leakage checks still strict.
3. If repeated forensic sweeps still show NO_EDGE, pivot to a different hypothesis class (stat-arb/cross-asset).

## Final Conclusion
- `BUG_FOUND_AND_FIXED`
