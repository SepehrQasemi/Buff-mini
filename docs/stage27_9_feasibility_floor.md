# Stage-27.9.2 Feasibility Floor Engine

## What Was Added
- New module: `src/buffmini/execution/feasibility_floor.py`
  - `calculate_min_risk_pct(equity, stop_distance_pct, min_notional, fee_roundtrip_pct, size_step)`
  - `calculate_min_equity(risk_pct, stop_distance_pct, min_notional)`
- `src/buffmini/execution/feasibility.py` now exposes alias fields used by Stage-27.9 reporting:
  - `min_risk_required`
  - `min_equity_required`
  - `stop_distance`
  - `fee_rt_pct`

## Reject Context Integration
- Stage-23 reject events now attach feasibility floor context for:
  - `SIZE_TOO_SMALL`
  - `POLICY_CAP_HIT`
  - `MARGIN_FAIL`
- Trace output includes:
  - `runs/<run_id>/trace/feasibility_analysis.csv`

## Purpose
- Quantify the true execution floor instead of only reporting reject labels.
- Separate alpha quality from feasibility constraints with deterministic math.
