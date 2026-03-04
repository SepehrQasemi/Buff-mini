# Stage-27.9.3 Adaptive Risk Bump

## Purpose
When live execution rejects an order with `SIZE_TOO_SMALL`, the engine now computes the minimum risk required for feasibility and attempts a deterministic one-step recovery.

## Logic
1. Trigger only in `live` mode and only for Stage-24 `risk_pct` sizing.
2. Compute `min_risk_required` from equity, stop distance, round-trip fee, min notional, and size step.
3. If `min_risk_required <= execution.risk_auto_bump.max_risk_cap`:
   - retry sizing with `risk_used = min_risk_required`;
   - accept only if sizing returns `VALID`.
4. Otherwise reject with `EXECUTION_INFEASIBLE_CAP`.

## Config
```yaml
execution:
  risk_auto_bump:
    enabled: true
    max_risk_cap: 0.20
```

## Trace Artifacts
- `runs/<run_id>/trace/risk_bump_events.csv`
- Stage24 sizing trace includes `risk_bump_applied` per attempt.

## Safety
- No entry/exit trigger changes.
- No relaxed live constraints beyond bounded risk bump.
- Deterministic behavior for identical seed/config/data.
