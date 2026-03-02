# Stage-23.1 Execution Reject Breakdown

## What was added
- Deterministic reject taxonomy in `src/buffmini/stage23/rejects.py`.
- Trace-level reject aggregation now writes:
  - `runs/<run_id>/trace/execution_reject_breakdown.json`
  - `runs/<run_id>/trace/execution_reject_events.csv`

## Reject reason taxonomy
- `SIZE_ZERO`
- `SIZE_TOO_SMALL`
- `STOP_TOO_CLOSE`
- `STOP_INVALID`
- `RR_INVALID`
- `SLIPPAGE_TOO_HIGH`
- `SPREAD_TOO_HIGH`
- `MARGIN_FAIL`
- `POSITION_CONFLICT`
- `DELAY_FAIL`
- `NO_FILL`
- `POLICY_CAP_HIT`
- `UNKNOWN`

## Artifact fields
- `total_orders_attempted`
- `total_orders_accepted`
- `total_orders_rejected`
- `reject_reason_counts`
- `reject_reason_rate`

The breakdown enforces accounting consistency:
- `accepted + rejected == attempted`
- Unknown reasons are normalized to `UNKNOWN` only when needed.

