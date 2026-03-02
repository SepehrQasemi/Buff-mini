# Stage-18 Report

## 1) What changed
- status: `PASS`

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage18.py --dry-run --seed 42`
- real-local: `python scripts/run_stage18.py --seed 42`

## 3) Validation gates & results
- run_id: `20260302_000508_f3399e6887fe_stage18`
- seed: `42`
- config_hash: `46824450e5e7e3ca`
- data_hash: `c9a171ee50947a7c`
- resolved_end_ts: `2026-02-26T09:00:00+00:00`
- trade_count: `0.0`
- trades_per_month: `0.0`
- exposure_ratio: `0.0`
- PF: `0.0`
- PF_raw: `0.0`
- expectancy: `0.0`
- exp_lcb: `-0.0003361223479059816`
- max_drawdown: `0.0`
- walkforward_executed_true_pct: `0.0`
- usable_windows_count: `0`
- mc_trigger_rate: `0.0`
- invalid_pct: `0.0`
- zero_trade_pct: `0.0`
- accepted_effects_count: `0`
- summary_hash: `d0e4cae2cbaf8eec`

## 4) Key metrics tables
- See JSON summary for full machine-readable values.

## 5) Failures + reasons
- no_accepted_conditional_effects

## 6) Next actions
- Stage-19: map accepted contexts into transition entry components.
- Keep falsification rules strict when sample sizes shrink.
