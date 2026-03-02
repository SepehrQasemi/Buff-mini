# Stage-19 Report

## 1) What changed
- status: `PASS`

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage19.py --dry-run --seed 42`
- real-local: `python scripts/run_stage19.py --seed 42`

## 3) Validation gates & results
- run_id: `20260302_020314_575b3f9deb39_stage19`
- seed: `42`
- config_hash: `46824450e5e7e3ca`
- data_hash: `01f14e86fc831288`
- resolved_end_ts: `2025-04-10T23:00:00+00:00`
- trade_count: `146.5`
- trades_per_month: `43.968320133388914`
- exposure_ratio: `0.6331249999999999`
- PF: `0.6782125489252823`
- PF_raw: `0.6782125489252823`
- expectancy: `-27.10868198211099`
- exp_lcb: `-39.94730251645879`
- max_drawdown: `0.4565748604651889`
- walkforward_executed_true_pct: `0.0`
- usable_windows_count: `0`
- mc_trigger_rate: `0.0`
- invalid_pct: `0.0`
- zero_trade_pct: `0.0`
- delta_exp_lcb_vs_baseline: `-39.94730251645879`
- summary_hash: `22bc7de8f2b4955b`

## 4) Key metrics tables
- See JSON summary for full machine-readable values.

## 5) Failures + reasons
- none

## 6) Next actions
- Stage-20: rank candidates with robust objective constraints.
- Keep transition score bounded to avoid signal spam.
