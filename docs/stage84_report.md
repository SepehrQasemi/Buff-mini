# Stage-84 Report

- status: `PARTIAL`
- execution_status: `EXECUTED`
- stage_role: `real_validation`
- validation_state: `SERIOUS_EDGE_CAMPAIGN_BLOCKED`
- mode: `evaluation`
- interpretation_allowed: `True`
- baseline_resolved_end_ts: `2026-03-03T10:00:00+00:00`
- assets: `['BTC/USDT', 'ETH/USDT']`
- mechanism_families: `['structure_pullback_continuation', 'failed_breakout_reversal', 'volatility_regime_transition', 'exhaustion_mean_reversion']`
- evaluated_assets: `0`
- blocked_assets: `2`
- edge_inventory_count: `0`
- campaign_outcome: `system_blocked_uninterpretable`

## Blocked Assets
- BTC/USDT: runtime_truth_blocked=`False` continuity_blocked=`True` runtime_truth_reason=`` continuity_reason=`gap_count=1,largest_gap_bars=2,max_gap_bars=0`
- ETH/USDT: runtime_truth_blocked=`False` continuity_blocked=`True` runtime_truth_reason=`` continuity_reason=`gap_count=1,largest_gap_bars=2,max_gap_bars=0`

## Candidate Classes
- promising_but_unproven: `0`
- rejected: `0`
- robust_candidate: `0`

## Failure Analysis
- blocked_continuity::gap_count=1,largest_gap_bars=2,max_gap_bars=0: `2`

- summary_hash: `1585d5d4e29a2b34`
