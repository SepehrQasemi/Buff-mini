# Stage-38 Logic Audit Report

## Hunt vs Engine Lineage
| metric | value |
| --- | ---: |
| raw_signal_count | 0.000000 |
| legacy_raw_signal_count | 87650.000000 |
| post_threshold_count | 0.000000 |
| post_cost_gate_count | 0.000000 |
| post_feasibility_count | 0.000000 |
| composer_signal_count | 0.000000 |
| engine_raw_signal_count | 0.000000 |
| final_trade_count | 0.000000 |

## Before vs After Evidence
| metric | before | after |
| --- | ---: | ---: |
| raw_signal_count | 87650.000000 | 0.000000 |
| composer_vs_engine_delta | 87650.000000 | 0.000000 |

## Collapse Point
- collapse_reason: `no_raw_candidates`
- contradiction_fixed: `True`
- root_cause: `Activation hunt counted NaN active-candidate cells as non-empty strings ('nan'), inflating raw_signal_count while engine final_signal stayed zero.`
- fix_summary: `Normalized active_candidates now maps NaN/None/'nan' to empty before raw-signal gating, and lineage now tracks composer_signal_count explicitly.`

## OI Short-Only Enforcement
- short_only_enabled: `True`
- timeframe: `1h`
- timeframe_allowed: `False`
- oi_active_runtime: `False`
- oi_non_null_rows: `0`

## Self-Learning Registry
- registry_path: `.`
- registry_rows: `0`
- elites_count: `0`
- dead_family_count: `0`
- failure_motif_tags_non_empty: `False`
