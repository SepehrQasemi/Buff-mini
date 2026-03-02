# Stage-23.3 Unified Eligibility Gate

## Objective
Replace stacked `context -> confirm -> riskgate` rejects with one deterministic eligibility score gate to remove redundant filtering while preserving risk discipline.

## Implementation
- Added `src/buffmini/stage23/eligibility.py`.
- Added unified API:
  - `evaluate_eligibility(features, regime, policy_snapshot, symbol, ts) -> eligible, score, reasons`
- Stage-23-enabled trace flow now evaluates eligibility once, then maps post-gate counts.

## Trace artifact
- `runs/<run_id>/trace/eligibility_trace.csv`
  - columns include `ts`, `symbol`, `score`, `threshold`, `eligible`, `reasons`, `regime`, `family`

## Determinism and bounds
- Eligibility score is clipped to `[0,1]`.
- Thresholds are deterministic and config-driven (`min_score_default`, optional per-regime overrides).
- Rejection reasons are explicit (`no_trend_confirmation`, `low_volatility`, `risk_policy_conflict`, fallback score threshold reason).

