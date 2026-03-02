# Stage-22 Report

## 1) What changed
- status: `PASS`

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage22.py --dry-run --seed 42`
- real-local: `python scripts/run_stage22.py --seed 42`

## 3) Validation gates & results
- run_id: `20260302_004522_9dafae0548fe_stage22`
- seed: `42`
- config_hash: `46824450e5e7e3ca`
- data_hash: `e246dd1e80636881`
- resolved_end_ts: `2026-02-26T09:00:00+00:00`
- trade_count: `4.5`
- trades_per_month: `0.11714936544093718`
- exposure_ratio: `0.0018982536066818527`
- PF: `2.4443039541787797`
- PF_raw: `2.4443039541787797`
- expectancy: `89.43501287682408`
- exp_lcb: `-12.57322044705004`
- max_drawdown: `0.029687818667060373`
- walkforward_executed_true_pct: `0.0`
- usable_windows_count: `0`
- mc_trigger_rate: `0.0`
- invalid_pct: `0.0`
- zero_trade_pct: `0.0`
- conflict_rate_pct: `0.04158079328922154`
- bias_alignment_rate_pct: `99.95841920671077`
- delta_exp_lcb_vs_baseline: `-12.57322044705004`
- summary_hash: `d4e52f7753d9a9f3`

## 4) Key metrics tables
- See JSON summary for full machine-readable values.

## 5) Failures + reasons
- none

## 6) Next actions
- Run Stage-15..22 master A/B summary.
- Inspect conflict mode differences (net/hedge/isolated) under same seed.
