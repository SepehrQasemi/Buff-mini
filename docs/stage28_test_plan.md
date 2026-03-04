# Stage-28 Test Plan

## Windowing (Stage-28.1)
- `tests/test_stage28_window_calendar_counts.py`
  - Verifies exact rolling window counts on a deterministic 48-month synthetic timeline.
  - Asserts 3m/1m produces 46 windows and 6m/1m produces 42 windows.
- `tests/test_stage28_window_calendar_no_overlap_bug.py`
  - Verifies monotonic window starts/ends and exact month-step progression.
  - Guards against off-by-one and malformed calendar boundaries.

## Notes
- All tests are offline and deterministic.
- Window generation uses UTC timestamps and deterministic month offsets.

## Context Discovery (Stage-28.2)
- `tests/test_stage28_context_mask_correctness.py`
  - Verifies context masks apply causally and scores stay in bounds.
- `tests/test_stage28_contextual_metrics_contract.py`
  - Verifies deterministic contextual metrics schema and repeatability.

## Budget Funnel (Stage-28.3)
- `tests/test_stage28_funnel_exploration_quota.py`
  - Ensures exploration floor is enforced.
- `tests/test_stage28_diversity_constraint.py`
  - Ensures diversity constraint limits near-duplicate candidates.

## WF/MC Usability (Stage-28.4)
- `tests/test_stage28_wf_triggers_when_usable.py`
  - Verifies usable contextual candidates trigger WF evaluation.
- `tests/test_stage28_mc_pooling_for_rare_contexts.py`
  - Verifies rare-context pooling behavior for MC preconditions.

## Feasibility Envelope (Stage-28.5)
- `tests/test_stage28_feasibility_envelope_math.py`
  - Verifies feasibility floor math and envelope aggregation.
- `tests/test_stage28_shadow_live_flags_present.py`
  - Verifies shadow-live feasibility flags are emitted.

## ML-lite Prioritizer (Stage-28.6)
- `tests/test_stage28_ml_ranker_does_not_change_search_space.py`
  - Verifies exploration quota is preserved with ML prioritization.
- `tests/test_stage28_ml_ranker_reproducible.py`
  - Verifies deterministic ranking with fixed seed.

## Policy Composer v2 (Stage-28.7)
- `tests/test_stage28_policy_v2_contract.py`
  - Verifies policy schema, per-context weights, and library export integration.
- `tests/test_stage28_policy_v2_determinism.py`
  - Verifies deterministic policy payload and composed signals.

## Orchestration & Report Schema (Stage-28.8)
- `tests/test_stage28_orchestrator_contract.py`
  - Verifies end-to-end dry-run contract and report artifacts.
- `tests/test_stage28_master_summary_schema.py`
  - Verifies required summary keys and schema integrity.

## Station Safety (Stage-28.9)
- `scripts/audit_stage28_station.py`
  - Verifies deterministic summary hashes across reruns, split-mode behavior, window consistency, leakage guard tests, and required docs.
