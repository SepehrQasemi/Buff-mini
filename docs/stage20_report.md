# Stage-20 Report

## 1) What changed
- status: `FAILED`

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage20.py --seed 42`
- real-local: `python scripts/run_stage20.py --seed 42`

## 3) Validation gates & results
- run_id: `20260302_002500_8ca55e995e03_stage20`
- seed: `42`
- config_hash: `46824450e5e7e3ca`
- data_hash: `a6129de40245d242`
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
- valid_candidates: `0`
- summary_hash: `309939e4a6e86d90`

## 4) Key metrics tables
- See JSON summary for full machine-readable values.

## 5) Failures + reasons
- no_valid_candidate_after_constraints

## 6) Next actions
- Stage-21: search-v2 bounded candidate generation and pruning.
- Keep objective constraints hard; never accept degenerate low-trade candidates.
