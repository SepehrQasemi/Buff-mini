# Stage-26 Policy Composer

## Purpose
Build a single conditional policy from rulelet-level evidence by context, with deterministic conflict resolution.

## Selection Rules
- Filter rows by `min_occurrences_per_context`.
- Filter by `min_trades_in_context`, except rows marked `RARE`.
- Rank by `exp_lcb` then `expectancy`.
- Keep `top_k` rulelets per context.

## Weighting
- Raw weight source: `max(0, exp_lcb)`.
- Clamp each weight to `[w_min, w_max]`.
- Normalize to sum to 1 per context.

## Conflict Modes
- `net`: use net weighted score sign.
- `hedge`: pick side with larger weighted magnitude.
- `isolated`: directional sign from net score, with isolated bookkeeping handled upstream.

## Trace Output
- `policy_trace.csv` includes per-bar context, long/short/net scores, final signal, and conflict mode.
