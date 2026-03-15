# Master Execution Report

## Baseline
- baseline_branch: `main`
- baseline_commit: `a93ec8f97318086b87386391f71de9754eec2f29`
- local_main_matches_origin_main: `True`
- baseline_readme_accurate: `True`

## Stage Statuses
- Stage-95: status=`SUCCESS`, execution_status=`EXECUTED`, validation_state=`LIVE_USEFULNESS_READY`, summary_hash=`64b3919bad947f8b`
- Stage-96: status=`SUCCESS`, execution_status=`EXECUTED`, validation_state=`CANONICAL_REPAIR_READY`, summary_hash=`09bea7197c96250a`
- Stage-97: status=`SUCCESS`, execution_status=`EXECUTED`, validation_state=`RELAXED_TO_STRICT_BRIDGE_READY`, summary_hash=`c141897474ab5101`
- Stage-98: status=`SUCCESS`, execution_status=`EXECUTED`, validation_state=`MECHANISM_SATURATION_READY`, summary_hash=`4404d7c3ad6cbb12`
- Stage-99: status=`SUCCESS`, execution_status=`EXECUTED`, validation_state=`CANDIDATE_QUALITY_ACCELERATION_READY`, summary_hash=`6d7c3ef6cec06fd5`
- Stage-100: status=`SUCCESS`, execution_status=`EXECUTED`, validation_state=`MULTISCOPE_TRUTH_CAMPAIGN_READY`, summary_hash=`6b096b5f5c3b90a9`
- Stage-101: status=`SUCCESS`, execution_status=`EXECUTED`, validation_state=`NULL_HYPOTHESIS_ATTACK_READY`, summary_hash=`16c3ee3d07489205`
- Stage-102: status=`SUCCESS`, execution_status=`EXECUTED`, validation_state=`ROBUSTNESS_RESCUE_ATTEMPTS_READY`, summary_hash=`68f405dd1e564aca`
- Stage-103: status=`SUCCESS`, execution_status=`EXECUTED`, validation_state=`FINAL_EDGE_VERDICT_READY`, summary_hash=`984f60aa2aa00934`

## Controlled Detectability
- proven: `True`
- signal_detection_rate: `0.857143`
- bad_control_rejection_rate: `1.0`
- synthetic_winner_recall: `1.0`
- false_negative_rate_on_known_good: `0.142857`

## Stage-95 Usefulness
- stage95b_recommended: `True`
- stage95b_applied: `True`
- dead_weight_family_count: `1`
- usefulness_delta: `{'useful_candidate_delta': 0, 'promising_delta': 0, 'mean_rank_score_delta': 0.051593, 'mean_near_miss_delta': 0.0, 'replay_death_fraction_delta': 0.0}`

## Stage-96 Canonical
- snapshot_id: `DATA_FROZEN_EVAL_v2`
- snapshot_hash: `18fed3f19383b259`
- strict_usable_rows: `6`

## Stage-100 Truth Campaign
- truth_counts: `{'replay_fragile_signal_only': 12, 'data_blocks_interpretation': 6}`
- tier1_symbols: `['BTC/USDT', 'ETH/USDT']`
- tier2_symbols: `[]`
- candidate_limit_per_scope: `1`

## Stage-101 Null Attack
- candidate_count_reviewed: `3`
- control_win_counts: `{'delayed_fake_signal': 3, 'momentum_baseline': 3, 'mean_reversion_baseline': 3}`
- candidate_beats_all_controls_count: `0`
- candidate_beats_majority_controls_count: `0`

## Stage-102 Rescue
- candidate_limit_reviewed: `1`
- classification_counts: `{'still_generator_weak': 1}`

## Stage-103 Final Edge Verdict
- final_edge_verdict: `GENERATOR_OR_SEARCH_FORMALISM_STILL_INSUFFICIENT`
- evidence_table: `[{'stage': '95', 'metric': 'dead_weight_family_count', 'value': 1}, {'stage': '96', 'metric': 'canonical_usable_rows', 'value': 6}, {'stage': '100', 'metric': 'replay_fragile_signal_rows', 'value': 12}, {'stage': '100', 'metric': 'data_block_rows', 'value': 6}, {'stage': '101', 'metric': 'candidates_beating_all_controls', 'value': 0}, {'stage': '101', 'metric': 'candidates_beating_majority_controls', 'value': 0}, {'stage': '102', 'metric': 'rescueable_count', 'value': 0}, {'stage': '102', 'metric': 'transfer_blocked_count', 'value': 0}, {'stage': '102', 'metric': 'still_generator_weak_count', 'value': 1}]`

## GitHub
- execution_branch: `codex/stage95-103-max-execution`
- execution_head: `659a562152bc4fadf5f04af446522bfc5a99cb33`
- PR number: `5`
- PR title: `Stage 95-103: live truth push and final edge verdict`
- PR url: `https://github.com/SepehrQasemi/Buff-mini/pull/5`
- PR state: `OPEN`
- PR merge_state_status: `BLOCKED`

## Main Protection
- required_pull_request_reviews: `True`
- required_approving_review_count: `1`
- allow_force_pushes: `False`
- allow_deletions: `False`

## Final Verdict
- final_verdict: `PARTIAL_REPAIR_MEANINGFUL`
- summary_hash: `d093655dd92ca542`
