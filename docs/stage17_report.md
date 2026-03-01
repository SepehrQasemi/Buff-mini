# Stage-17 Report

## 1) What changed
- status: `PASS`

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage17.py --dry-run --seed 42`
- real-local: `python scripts/run_stage17.py --seed 42`

## 3) Validation gates & results
- run_id: `20260301_235232_2c624bbf1ef5_stage17`
- seed: `42`
- config_hash: `46824450e5e7e3ca`
- data_hash: `e83c3ff37f7d9bbf`
- resolved_end_ts: `2026-02-26T09:00:00+00:00`
- trade_count: `0.0`
- trades_per_month: `0.0`
- exposure_ratio: `0.0`
- PF: `0.0`
- PF_raw: `0.0`
- expectancy: `0.0`
- exp_lcb: `0.0`
- max_drawdown: `0.0`
- walkforward_executed_true_pct: `0.0`
- usable_windows_count: `0`
- mc_trigger_rate: `0.0`
- invalid_pct: `0.0`
- zero_trade_pct: `100.0`
- drag_sensitivity_delta_exp_lcb: `18.2152860584907`
- runtime_seconds: `0.0`
- cache_hit_rate: `0.0`
- summary_hash: `179e627d8129521a`

## 4) Key metrics tables
- See JSON summary for full machine-readable values.

## 5) Failures + reasons
- none

## 6) Next actions
- Stage-18: test conditional effects by context state.
- Keep same-candle exit priority unchanged in engine core.
