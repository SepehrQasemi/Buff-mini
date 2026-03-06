# Stage-38 Test Hardening Report

## New Guardrails Added
- `tests/test_stage38_activation_hunt_to_engine_trace.py`
  - Guards the NaN/"nan" raw-signal inflation regression.
  - Verifies explicit composer collapse reason classification.
- `tests/test_stage38_oi_short_only_enforced.py`
  - Verifies OI short-only masking on `1h`.
  - Verifies OI remains usable on sub-hour (`15m`) when data exists.
- `tests/test_stage38_oi_usage_report_matches_runtime.py`
  - Ensures OI runtime usage report matches feature-builder runtime state.
- `tests/test_stage38_self_learning_registry_population.py`
  - Ensures zero-trade cases still populate registry rows with failure motifs.
  - Verifies deterministic elite-flag persistence.
- `tests/test_stage38_policy_gate_trace.py`
  - Exercises trace lineage counters from Stage-28 artifacts.
  - Adds contradiction guard by asserting composer count equals engine raw count when signals exist.
- `tests/test_stage38_end_to_end_summary_schema.py`
  - Enforces Stage-38 master summary schema contract.

## Logical Edges Now Covered
- Activation-hunt -> engine lineage consistency.
- Composer/policy handoff count consistency.
- OI short-only enforcement and reporting consistency.
- Self-learning registry write path under zero-trade failure.
- Stage-38 summary contract stability.

## Remaining Gaps
- Full long-runtime Stage-28/37 replay under all market snapshots remains expensive and is not run in unit tests.
- UI-triggered subprocess orchestration paths are validated indirectly through artifact-based trace tests, not full Streamlit integration tests.

## Why These Gaps Remain
- Deterministic CI constraints and runtime cost make exhaustive snapshot matrix replay impractical.
- The highest-risk logical transitions are now protected by focused deterministic tests that fail fast on regressions.
