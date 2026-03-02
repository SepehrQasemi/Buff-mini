# Stage-22 Report

## 1) What changed
- status: `FAILED`

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage22.py --dry-run --seed 42`
- real-local: `python scripts/run_stage22.py --seed 42`

## 3) Validation gates & results
- run_id: `20260302_020319_eeb8b7fc994b_stage22`
- seed: `42`
- config_hash: `46824450e5e7e3ca`
- data_hash: `8ae821cb0c46bff5`
- resolved_end_ts: `2025-04-10T23:00:00+00:00`
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
- conflict_rate_pct: `0.0`
- bias_alignment_rate_pct: `100.0`
- delta_exp_lcb_vs_baseline: `0.0`
- summary_hash: `60c28f0241d252c5`

## 4) Key metrics tables
- See JSON summary for full machine-readable values.

## 5) Failures + reasons
- mtf_effect_not_measurable
- truth:no_trades_executed

## 6) Next actions
- Run Stage-15..22 master A/B summary.
- Inspect conflict mode differences (net/hedge/isolated) under same seed.
