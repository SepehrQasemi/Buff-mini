# Stage-23.6.2 Sizing Integrity Fix

## What changed
- Added deterministic quantity rounding helper: `round_qty_to_step(qty, step, mode)`.
- Added Stage-23 sizing controls:
  - `evaluation.stage23.sizing_fix_enabled`
  - `evaluation.stage23.sizing.qty_rounding_default`
  - `evaluation.stage23.sizing.qty_rounding_on_min_notional_bump`
  - `evaluation.stage23.sizing.allow_single_step_ceil_rescue`
  - `evaluation.stage23.sizing.ceil_rescue_max_overage_steps`
- Extended `evaluation.stage23.order_builder` with:
  - `min_trade_qty`
  - `qty_step`

## Behavior
- Min-notional bump now attempts to preserve executable quantity using deterministic rounding.
- If floor-rounding would produce zero while quantity is positive:
  - legacy (`sizing_fix_enabled=false`): can reject as `SIZE_ZERO`
  - fixed mode (`sizing_fix_enabled=true`): attempts one-step ceil rescue (bounded); if unsafe, rejects explicitly.
- No silent shrink-to-zero in fixed mode.

## Safety
- Caps and margin checks remain enforced.
- No randomness introduced; behavior is deterministic for the same inputs.
