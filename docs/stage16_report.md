# Stage-16 Report

## 1) What changed
- status: `PASS`

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage16.py --dry-run --seed 42`
- real-local: `python scripts/run_stage16.py --seed 42`

## 3) Validation gates & results
- run_id: `20260302_020307_a3059cc70f35_stage15`
- seed: `42`
- config_hash: `46824450e5e7`
- data_hash: `a60fb78eeab75168`
- resolved_end_ts: `2025-04-10T23:00:00+00:00`
- classic_trade_count: `10.0`
- alpha_trade_count: `0.0`
- classic_exp_lcb: `-106.14338942335678`
- alpha_exp_lcb: `0.0`
- max_state_share_pct: `78.08333333333334`
- summary_hash: `e2e4ccc85b65b120`

## 4) Key metrics tables
- See JSON summary for full machine-readable values.

## 5) Failures + reasons
- none

## 6) Next actions
- Stage-17: evaluate exit-v2 variants with fixed entries.
- Use transition matrix to weight soft context routing only.
