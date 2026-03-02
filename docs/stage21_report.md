# Stage-21 Report

## 1) What changed
- status: `FAILED`

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage21.py --dry-run --seed 42`
- real-local: `python scripts/run_stage21.py --seed 42`

## 3) Validation gates & results
- run_id: `20260302_020317_78e44129c6f9_stage21`
- seed: `42`
- config_hash: `46824450e5e7e3ca`
- data_hash: `35bebbef4706684b`
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
- invalid_pct: `100.0`
- zero_trade_pct: `0.0`
- evaluated_count: `90`
- pruned_count: `90`
- runtime_seconds: `0.04822229999990668`
- cache_hit_rate: `0.0`
- summary_hash: `c745e884dc574dd0`

## 4) Key metrics tables
- See JSON summary for full machine-readable values.

## 5) Failures + reasons
- search_returned_empty_candidate_set

## 6) Next actions
- Stage-22: apply MTF policy with strict no-leak alignment.
- Inspect prune_reasons to adjust search-space quality.
