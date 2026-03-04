# Stage-27.9.1 Research/Live Execution Separation

## What Changed
- Added explicit `evaluation.modes.research` and `evaluation.modes.live` behavior controls.
- Research mode now keeps cost realism (`fees`, `slippage`, `spread`) but does not hard-reject on:
  - `min_notional`
  - `size_step` / precision
  - margin/cap feasibility checks
- Live mode keeps exchange-like execution rules.

## New Config Surface
- `evaluation.modes.research.min_notional_override`
- `evaluation.modes.research.ignore_exchange_precision`
- `evaluation.modes.research.enforce_margin_caps`
- `evaluation.modes.research.enforce_min_notional`
- `evaluation.modes.research.enforce_size_step`
- `evaluation.modes.live.use_exchange_rules`

## Trace Artifact
- Research-mode accepted trades that would fail live checks are flagged:
  - `runs/<run_id>/trace/research_infeasible_flags.csv`
- Each row includes timestamp/symbol/side/reason and live rule values.

## Why
- Discovery should evaluate signal quality without being choked by exchange minimums.
- Live feasibility remains auditable through shadow-live flags instead of silent pruning.
