# Stage-23.6.1 Sizing Trace Diagnostics

## What was added
- Per-order attempt sizing trace records in `src/buffmini/stage23/order_builder.py`.
- Trace artifacts written by `src/buffmini/forensics/signal_flow.py`:
  - `runs/<run_id>/trace/sizing_trace.csv`
  - `runs/<run_id>/trace/sizing_trace_summary.json`

## Trace schema (core fields)
- `ts`, `symbol`, `side`
- `price`, `stop_price`, `tp_price`
- `raw_size`, `capped_size`
- `min_notional`, `min_trade_qty`, `qty_step`
- `bumped_to_min_notional`
- `rounded_size_before`, `rounded_size_after`, `rounding_mode_used`
- `final_notional`
- `decision`, `reject_reason`, `reject_details`

## Interpretation
- `raw_size` is the pre-adjustment quantity estimate.
- `rounded_size_after` is the final quantity used for accept/reject.
- `SIZE_ZERO` is reserved for cases where size was positive before rounding and became zero after rounding (target bug class for 23.6.2+).
- `sizing_trace_summary.json` provides aggregate diagnostics (`attempted`, `accepted`, `rejected`, quantiles, zero-size counts, and reject reason counts).
