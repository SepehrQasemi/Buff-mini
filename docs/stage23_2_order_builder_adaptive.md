# Stage-23.2 Adaptive Order Builder

## Objective
Reduce unnecessary order rejection by adapting stop distance, risk/reward fallback, and minimum notional handling while preserving risk caps and determinism.

## Implementation
- Added `src/buffmini/stage23/order_builder.py`.
- New config section:
  - `evaluation.stage23.order_builder.*`
  - `evaluation.stage23.execution.*`
- Integrated into trace pipeline when `evaluation.stage23.enabled=true`.

## Behavior
- Tight stops are clamped using max of ATR-based and bps-based minimum distance.
- Invalid RR can fall back to configured exit mode; otherwise reject with `RR_INVALID`.
- Small notionals can be bumped to configured minimum when allowed.
- Margin/cap failure can trigger deterministic size reduction before final reject.

## Output
- Accepted signal stream for backtest routing.
- Structured order rows (`VALID`/`INVALID` semantics).
- Structured reject events with known reason taxonomy.

