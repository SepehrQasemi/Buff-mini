# Stage-24 Design

## Scope
Stage-24 adds deterministic sizing realism and capital-level simulation without changing alpha logic.

## Sizing modes
- `risk_pct` (primary): size from risk budget per trade.
- `alloc_pct` (secondary): size from fixed equity allocation percentage.

## Risk ladder
Base risk (when no user override):

`base_risk = clamp(r_min, r_max, r_ref * (e_ref / equity)^k)`

Then apply safety multipliers:
- Drawdown clamps (`dd_soft`, `dd_hard`) with configured multipliers.
- Losing-streak clamps (`losing_streak_soft`, `losing_streak_hard`) with configured multipliers.

Final risk:

`risk_used = clamp(r_min, r_max, base_risk * dd_mult * streak_mult)`

## Cost-aware notional
For `risk_pct` mode:

`notional = (equity * risk_pct_used) / (stop_distance_pct + cost_rt_pct)`

This includes cost in denominator so notional shrinks as costs rise.

## Why this helps choke diagnosis
Small-cap failures become explicit through deterministic reject reasons:
- `SIZE_TOO_SMALL`
- `POLICY_CAP_HIT`
- `MARGIN_FAIL`

Stage-24 reporting compares capital buckets (100/1k/10k/100k) and surfaces where minimum notional or caps block execution.
