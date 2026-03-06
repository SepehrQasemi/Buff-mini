# Stage-37 Master Report

## Changes
- Added activation-hunt reject-chain diagnostics with threshold sensitivity.
- Expanded free derivatives feature families (funding/taker/long-short) and OI short-horizon mode.
- Added failure-aware self-learning registry and deterministic elite/pruning logic.

## Data Families
- funding available: `True`
- taker available: `True`
- long_short available: `True`
- oi_short_only_mode: `False`

## Engine
- baseline_run_id: `20260306_160459_2a029423a621_stage28`
- upgraded_run_id: `20260306_144056_4b655c81c2a5_stage28`
- delta_activation_rate: `0.000000`
- delta_trade_count: `0.000000`
- delta_live_exp_lcb: `0.000000`

## Stability
- executed_seed_count: `0`
- note: `Not promising on seed-42; skipped extra seeds to avoid waste.`

## Verdict
- verdict: `DATA_IMPROVED_BUT_NO_EDGE`
- biggest_remaining_bottleneck: `cost_drag_vs_signal`
- next_cheapest_action: `Tune Stage-28/37 activation and cost-gate settings per family using stage37_activation_hunt_report.md`
