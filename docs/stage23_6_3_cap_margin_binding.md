# Stage-23.6.3 Explicit Cap/Margin Binding

## Objective
Stop silent quantity shrink-to-zero and make binding constraints explicit.

## Changes
- Sizing pipeline now tracks `cap_binding` in `sizing_trace`:
  - `policy_cap`
  - `margin`
- If a bound path pushes effective size below executable quantity:
  - `POLICY_CAP_HIT` for policy-cap binding
  - `MARGIN_FAIL` for margin binding
- `SIZE_ZERO` is no longer used as a fallback for cap/margin binding in fixed mode.

## Diagnostics
- `runs/<run_id>/trace/sizing_trace.csv` includes `cap_binding`.
- `runs/<run_id>/trace/sizing_trace_summary.json` includes:
  - `cap_binding_reject_count`
  - `reject_reason_counts`

## Integrity
- The breakdown identity remains enforced:
  - `attempted = accepted + rejected`
