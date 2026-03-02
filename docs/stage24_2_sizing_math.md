# Stage-24.2 Sizing Math

## Risk% mode
Inputs:
- `equity`
- `risk_pct_used`
- `stop_distance_pct`
- `cost_rt_pct`

Formula:

`denom = stop_distance_pct + cost_rt_pct`

`notional_raw = equity * risk_pct_used / denom`

Then apply constraints deterministically:
- cap by `max_notional_pct_of_equity * equity`
- min-notional bump (if enabled and feasible)
- otherwise explicit `SIZE_TOO_SMALL` reject

## Allocation% mode

`notional_raw = equity * alloc_pct`

Same min-notional/cap rules as Risk% mode.

## Risk ladder
If `risk_pct_user` is set, it is used (clamped to `[r_min, r_max]`).

Else:

`base = clamp(r_min, r_max, r_ref * (e_ref / equity)^k)`

Then:
- drawdown multiplier (`dd_soft` / `dd_hard`)
- losing-streak multiplier (`losing_streak_soft` / `losing_streak_hard`)

Final:

`risk_used = clamp(r_min, r_max, base * dd_mult * streak_mult)`

## Determinism
All Stage-24 sizing outputs are pure deterministic transforms of input values and config; no randomness is used.
