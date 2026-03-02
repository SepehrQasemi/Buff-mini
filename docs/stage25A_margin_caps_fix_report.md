# Stage-25A Margin/Caps Fix Report

## Scope
- Stage-25A focuses on execution feasibility correctness only:
  - canonical cap/margin model
  - deterministic ordering (caps before margin)
  - equity-tier anomaly elimination
  - research/live constraint split (added in 25.3)

## Stage-25.1 Changes
- Added canonical module: `src/buffmini/execution/margin_model.py`
  - `compute_margin_required`
  - `apply_exposure_caps`
  - `is_trade_feasible`
- Integrated Stage-23 order builder with canonical cap/margin ordering for Stage-24 path:
  1. start from desired notional
  2. apply policy caps
  3. compute margin required
  4. check feasibility against margin allocation limit
- Preserved legacy Stage-23 behavior when Stage-24 is disabled (backward compatibility).
- Added sizing trace fields for auditability:
  - `max_allowed_notional`
  - `margin_required`
  - `margin_limit`
- Added invariant tests:
  - `tests/test_margin_model_invariants.py`

## Stage-25.2 Evidence (Stage-24 anomaly fixed)
- Pre-fix snapshot: `docs/stage25A_stage24_prefx_snapshot.json`
- Post-fix rerun: `docs/stage24_report_summary.json`

### Equity-tier comparison (dry-run, seed=42)
| initial_equity | pre trade_count | pre invalid_pct | pre top_reason | post trade_count | post invalid_pct | post top_reason |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 100 | 32 | 0.0 | VALID | 32 | 0.0 | VALID |
| 1,000 | 32 | 0.0 | VALID | 32 | 0.0 | VALID |
| 10,000 | 0 | 100.0 | MARGIN_FAIL | 32 | 0.0 | VALID |
| 100,000 | 0 | 100.0 | MARGIN_FAIL | 32 | 0.0 | VALID |

- Added regression test:
  - `tests/test_stage24_equity_scaling_no_pathological_margin_fail.py`
- Result: pathological high-equity `MARGIN_FAIL` is removed in offline dry-run replay.

## Invariants Enforced
- Equity scaling invariant for proportional sizing:
  - If equity scales by `K`, notional and margin scale by `K`.
  - Feasibility does not flip feasible->infeasible solely due to larger equity.
- Ordering invariant:
  - cap binding is reported as `POLICY_CAP_HIT` before margin failure checks.

## Notes
- Stage-25.3 adds research/live constraints and shadow-live feasibility accounting.

## Stage-25.3 Research vs Live Constraints Split
- Added config namespace: `evaluation.constraints`:
  - `mode`: `live|research`
  - `live`: `min_trade_notional`, `min_trade_qty`, `qty_step`
  - `research`: `min_trade_notional`, `min_trade_qty`, `qty_step`
- Order builder now enforces active constraints by mode while preserving all existing risk controls.
- In `research` mode, every accepted trade is shadow-checked against `live` constraints and counted.

### Shadow-live evidence (dry-run trace)
- run_id: `20260302_212014_26d21b14c3d4_stage15_9_trace`
- artifact: `runs/20260302_212014_26d21b14c3d4_stage15_9_trace/trace/shadow_live_summary.json`
- summary:
  - `research_accepted_count = 8`
  - `live_pass_count = 5`
  - `research_accepted_but_live_rejected_count = 3`
  - `live_reject_reason_counts.SIZE_TOO_SMALL = 3`

### Tests added
- `tests/test_constraints_mode_split.py`
  - research mode accepts smaller notionals and reports shadow-live rejects.
  - live mode rejects the same small notionals with `SIZE_TOO_SMALL`.
