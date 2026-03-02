# Stage-15 Report

## 1) What changed
- status: `PASS`

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage15.py --dry-run --seed 42`
- real-local: `python scripts/run_stage15.py --seed 42`

## 3) Validation gates & results
- run_id: `20260302_020305_5ce87c9aa3df_stage15`
- seed: `42`
- config_hash: `46824450e5e7`
- data_hash: `a60fb78eeab75168`
- resolved_end_ts: `2025-04-10T23:00:00+00:00`
- classic_trade_count: `10.0`
- alpha_trade_count: `0.0`
- trade_count: `10.0`
- classic_exp_lcb: `-106.14338942335678`
- alpha_exp_lcb: `0.0`
- delta_exp_lcb: `106.14338942335678`
- activation_pct_not_neutral: `100.0`
- summary_hash: `3f8b53ec78f8944a`

## 4) Key metrics tables
- See JSON summary for full machine-readable values.

## 5) Failures + reasons
- none

## 6) Next actions
- Stage-16: add context persistence and no-leak checks.
- Use A/B runner hashes to detect no-op regressions.
