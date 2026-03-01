# Stage-16 Report

## 1) What changed
- status: `PASS`

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage16.py --dry-run --seed 42`
- real-local: `python scripts/run_stage16.py --seed 42`

## 3) Validation gates & results
- run_id: `20260301_233850_fc5bd541a924_stage15`
- seed: `42`
- config_hash: `46824450e5e7`
- data_hash: `9ee4982b93e3ac04`
- resolved_end_ts: `2026-02-26T09:00:00+00:00`
- classic_trade_count: `14.0`
- alpha_trade_count: `0.0`
- classic_exp_lcb: `30.099127775582197`
- alpha_exp_lcb: `0.0`
- max_state_share_pct: `70.60057128394259`
- summary_hash: `35795e411b0ccadb`

## 4) Key metrics tables
- See JSON summary for full machine-readable values.

## 5) Failures + reasons
- none

## 6) Next actions
- Stage-17: evaluate exit-v2 variants with fixed entries.
- Use transition matrix to weight soft context routing only.
