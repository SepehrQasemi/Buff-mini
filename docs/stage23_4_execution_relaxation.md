# Stage-23.4 Execution Realism Relaxation

## Objective
Reduce hard execution rejection where a safe fallback exists, without bypassing risk caps.

## Implemented behavior
- **Margin failure handling**:
  - deterministic size-reduction loop (bounded by `max_size_reduction_steps`)
  - rejects with `MARGIN_FAIL` if still above cap
- **Slippage handling**:
  - soft threshold: reduce size
  - hard threshold: reject with `SLIPPAGE_TOO_HIGH`
- **Spread handling**:
  - reject with `SPREAD_TOO_HIGH` if spread exceeds hard cap
- **Partial fill support**:
  - deterministic fill ratio from liquidity proxy
  - clipped by `partial_fill_min_ratio`

## New trace evidence
- `runs/<run_id>/trace/execution_adjustments.csv`
  - logs each size-reduction attempt and rationale
- `runs/<run_id>/trace/execution_reject_events.csv`
  - rejects with explicit reason and details

## Determinism
All adjustment logic is deterministic (no random branch, no stochastic fill path).

