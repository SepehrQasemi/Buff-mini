# Stage-38 Master Report

## What Was Broken Before Stage-38
- Activation-hunt counts were inflated by NaN active-candidate parsing and diverged from engine replay zero-signal reality.
- OI short-only behavior existed in parts of the pipeline but runtime/report consistency was not explicit.
- Self-learning memory could remain empty on zero-trade runs, preventing failure-aware evolution.

## What Was Fixed
- Added deterministic runtime trace from entrypoint to report artifacts.
- Fixed hunt raw-signal parsing and added explicit composer lineage counts.
- Added strict OI short-horizon runtime usage metadata and consistency checks.
- Ensured self-learning registry persists failure motifs and elite flags even in zero-trade runs.

## Logic Consistency
- stage28_run_id: `20260306_152631_2a029423a621_stage28`
- trace_hash: `4e4d501fbca6d698`
- contradiction_fixed: `True`
- collapse_reason: `no_raw_candidates`

## OI Short-Only
- short_only_enabled: `True`
- timeframe: `1h`
- timeframe_allowed: `False`
- oi_active_runtime: `False`

## Self-Learning Memory
- registry_rows: `1`
- elites_count: `1`
- dead_family_count: `1`
- failure_motif_tags_non_empty: `True`

## Engine State
- engine_raw_signal_count: `0`
- final_trade_count: `0.000000`
- still_no_edge: `True`

## Remaining Bottleneck
- biggest_remaining_bottleneck: `cost_drag_vs_signal`
- next_cheapest_high_confidence_action: `Increase upstream candidate signal quality (family/context generation), then rerun Stage-37/38 lineage checks.`

## Verdict
- `SELF_LEARNING_NOW_REAL_BUT_SIGNAL_WEAK`
- summary_hash: `1024d9be2719caa8`
