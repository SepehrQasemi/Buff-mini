# Stage-39..43 Master Report

## Stage Changes
- Stage-39: widened context grammar and layered candidate funnel (A/B/C).
- Stage-40: tradability labels + two-stage objective routing.
- Stage-41: derivatives family contribution metrics with OI short-only runtime guard.
- Stage-42: failure-aware self-learning memory, motifs, and mutation guidance.
- Stage-43: runtime instrumentation, baseline-vs-upgraded comparison, and 5-seed validation gate.

## Upstream Signal Generation
- baseline_engine_raw_signal_count: `0`
- upgraded_raw_candidate_count: `24`
- shortlisted_count: `5`

## Tradability Objective
- stage_a_survivors: `3`
- stage_b_survivors: `3`
- strongest_bottleneck_step: `stage_a_activation`

## Derivatives Contributions
- funding_available: `True`
- taker_available: `True`
- long_short_available: `True`
- oi_short_only_mode_enabled: `True`

## Self-Learning 2.0
- registry_rows_v2: `1`
- elite_count: `1`
- mutate_next: `widen_context_and_expand_grammar`

## Engine Performance (Baseline vs Upgraded)
- baseline_run_id: `20260306_160459_2a029423a621_stage28`
- upgraded_run_id: `20260306_144056_4b655c81c2a5_stage28`
- delta_raw_signal_count: `0.000000`
- delta_activation_rate: `0.000000`
- delta_trade_count: `0.000000`
- delta_live_best_exp_lcb: `0.000000`

## Runtime
- slowest_phase: `replay_backtest`
- replay_backtest_seconds: `2361.551806`

## 5-Seed Validation
- skipped: `True`
- executed_seed_count: `0`
- note: `Skipped 5-seed validation because upgraded seed-42 remained fully dead (zero raw signals and zero trades).`

## Final Verdict
- verdict: `RAW_SIGNAL_IMPROVED_BUT_NO_EDGE`
- biggest_remaining_bottleneck: `cost_drag_vs_signal`
- next_cheapest_high_confidence_action: `Increase upstream tradable signal families and tune Stage-A acceptance while preserving cost realism.`
- summary_hash: `a715969f47e950fc`
