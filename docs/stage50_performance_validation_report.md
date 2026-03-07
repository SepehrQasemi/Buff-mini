# Stage-50 Performance Validation Report

- status: `SUCCESS`
- baseline_runtime_seconds: `2274.706470`
- upgraded_runtime_seconds: `2361.551806`
- delta_runtime_seconds: `86.845335`
- slowest_phase: `replay_backtest`
- baseline_raw_signals: `0`
- upgraded_raw_signals: `0`
- baseline_trade_count: `0.000000`
- upgraded_trade_count: `0.000000`
- live_best_exp_lcb_before: `0.000000`
- live_best_exp_lcb_after: `0.000000`
- promising: `False`

## Runtime By Phase
- analyst_brain_part1: 0.000001s
- analyst_brain_part2: 0.000000s
- config_load: 0.051329s
- data_load: 0.001347s
- extras_alignment: 0.000000s
- monte_carlo: 0.000000s
- ranker_scoring: 0.000000s
- replay_backtest: 2361.551806s
- report_generation: 0.000007s
- setup_generation: 0.000001s
- stage_a_tradability_filter: 0.000000s
- stage_b_robustness: 0.000000s
- walkforward: 0.000000s

## Notes
- Upgraded path remains dead on seed-42 (no raw signals, no trades, no LCB improvement).
- 5-seed run skipped to avoid waste; explicit skip evidence written to Stage-50 5-seed summary.

- summary_hash: `1f65f53163ca34f7`
