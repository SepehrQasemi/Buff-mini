# Stage-25A Margin/Caps Fix Report

## Scope
- Stage-25A focuses on execution feasibility correctness only:
  - canonical cap/margin model
  - deterministic ordering (caps before margin)
  - invariant tests for equity scaling

## Stage-25.1 Changes
- Added canonical module: `src/buffmini/execution/margin_model.py`
  - `compute_margin_required`
  - `apply_exposure_caps`
  - `is_trade_feasible`
- Integrated Stage-23 order builder with canonical cap/margin ordering:
  1. start from desired notional
  2. apply policy caps
  3. compute margin required
  4. check feasibility against margin allocation limit
- Added sizing trace fields for auditability:
  - `max_allowed_notional`
  - `margin_required`
  - `margin_limit`
- Added invariant tests:
  - `tests/test_margin_model_invariants.py`

## Invariants Enforced
- Equity scaling invariant for proportional sizing:
  - If equity scales by `K`, notional and margin scale by `K`.
  - Feasibility does not flip feasible->infeasible solely due to larger equity.
- Ordering invariant:
  - cap binding is reported as `POLICY_CAP_HIT` before any margin failure check.

## Pending (covered in 25.2/25.3)
- Stage-24 equity-tier anomaly before/after evidence.
- Research vs Live constraint split and shadow-live reporting.
