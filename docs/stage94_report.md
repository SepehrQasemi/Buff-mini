# Stage-94 Report

- status: `SUCCESS`
- validation_state: `EDGE_INVENTORY_READY`
- symbols: `['BTC/USDT', 'ETH/USDT']`
- timeframes: `['1h', '4h']`
- mechanism_families: `['liquidity_sweep_reversal', 'structure_pullback_continuation', 'failed_breakout_reversal', 'volatility_regime_transition', 'exhaustion_mean_reversion']`
- campaign_outcome: `data_blocked_or_scope_blocked`
- classification: `system_blocked_uninterpretable`

## Candidate Class Counts
- `{"class": "robust_candidate", "count": 0}`
- `{"class": "promising_but_unproven", "count": 0}`
- `{"class": "rejected", "count": 0}`

## Mechanism Inventory
- none

## Regime Map
- none

## Blocked Scope Rows
- `{"blocked_reason": "gap_count=1,largest_gap_bars=2,max_gap_bars=0", "mode_summary": {"blocked_reasons": [], "campaign_memory_cold_start": true, "canonical_status": "CANONICAL", "config_hash_effective": "e26488a1d660", "continuity_status": "STRICT", "coverage_rows": [], "data_hash_status": "PRESENT", "data_scope_hash": "0e5aa94981778527", "effective_values": {"budget_mode_selected": "validate", "deterministic_sorting": true, "fail_on_gap": true, "frozen_research_mode": true, "require_resolved_end_ts": true, "strict_continuity": true}, "evaluation_mode": true, "interpretation_allowed": true, "mode": "evaluation", "resolved_end_ts": "2026-03-03T10:00:00+00:00", "resolved_end_ts_auto_pinned": false, "resolved_end_ts_status": "PINNED", "run_type": "evaluation", "seed_bundle": [11, 19, 23, 29, 31], "summary_hash": "975fac390da79561", "validation_state": "EVALUATION_READY"}, "symbol": "BTC/USDT", "timeframe": "1h"}`
- `{"blocked_reason": "gap_count=1,largest_gap_bars=1,max_gap_bars=0", "mode_summary": {"blocked_reasons": [], "campaign_memory_cold_start": true, "canonical_status": "CANONICAL", "config_hash_effective": "4a0fb345edff", "continuity_status": "STRICT", "coverage_rows": [], "data_hash_status": "PRESENT", "data_scope_hash": "85e2e9bf2ad2f054", "effective_values": {"budget_mode_selected": "validate", "deterministic_sorting": true, "fail_on_gap": true, "frozen_research_mode": true, "require_resolved_end_ts": true, "strict_continuity": true}, "evaluation_mode": true, "interpretation_allowed": true, "mode": "evaluation", "resolved_end_ts": "2026-03-03T04:00:00+00:00", "resolved_end_ts_auto_pinned": false, "resolved_end_ts_status": "PINNED", "run_type": "evaluation", "seed_bundle": [11, 19, 23, 29, 31], "summary_hash": "708dbf6920299f90", "validation_state": "EVALUATION_READY"}, "symbol": "BTC/USDT", "timeframe": "4h"}`
- `{"blocked_reason": "gap_count=1,largest_gap_bars=2,max_gap_bars=0", "mode_summary": {"blocked_reasons": [], "campaign_memory_cold_start": true, "canonical_status": "CANONICAL", "config_hash_effective": "e26488a1d660", "continuity_status": "STRICT", "coverage_rows": [], "data_hash_status": "PRESENT", "data_scope_hash": "0e5aa94981778527", "effective_values": {"budget_mode_selected": "validate", "deterministic_sorting": true, "fail_on_gap": true, "frozen_research_mode": true, "require_resolved_end_ts": true, "strict_continuity": true}, "evaluation_mode": true, "interpretation_allowed": true, "mode": "evaluation", "resolved_end_ts": "2026-03-03T10:00:00+00:00", "resolved_end_ts_auto_pinned": false, "resolved_end_ts_status": "PINNED", "run_type": "evaluation", "seed_bundle": [11, 19, 23, 29, 31], "summary_hash": "975fac390da79561", "validation_state": "EVALUATION_READY"}, "symbol": "ETH/USDT", "timeframe": "1h"}`
- `{"blocked_reason": "gap_count=1,largest_gap_bars=1,max_gap_bars=0", "mode_summary": {"blocked_reasons": [], "campaign_memory_cold_start": true, "canonical_status": "CANONICAL", "config_hash_effective": "4a0fb345edff", "continuity_status": "STRICT", "coverage_rows": [], "data_hash_status": "PRESENT", "data_scope_hash": "85e2e9bf2ad2f054", "effective_values": {"budget_mode_selected": "validate", "deterministic_sorting": true, "fail_on_gap": true, "frozen_research_mode": true, "require_resolved_end_ts": true, "strict_continuity": true}, "evaluation_mode": true, "interpretation_allowed": true, "mode": "evaluation", "resolved_end_ts": "2026-03-03T04:00:00+00:00", "resolved_end_ts_auto_pinned": false, "resolved_end_ts_status": "PINNED", "run_type": "evaluation", "seed_bundle": [11, 19, 23, 29, 31], "summary_hash": "708dbf6920299f90", "validation_state": "EVALUATION_READY"}, "symbol": "ETH/USDT", "timeframe": "4h"}`

## Transfer Map
- `{"failure_diagnostics": {}, "regime_portability_map": [], "summary_hash": "35fe542f5ca713e0", "transfer_class_counts": {}}`

## Failure Map
- `{"count": 2, "failure": "blocked::gap_count=1,largest_gap_bars=2,max_gap_bars=0"}`
- `{"count": 2, "failure": "blocked::gap_count=1,largest_gap_bars=1,max_gap_bars=0"}`

- summary_hash: `b3d6124eed197044`
