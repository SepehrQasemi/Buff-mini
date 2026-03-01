# Stage-13.5 Combined Composer Report

## 1) What changed
- Ran family-alone, pairwise, and all-family composer matrix.
- Added interaction analysis: overlap, return-correlation, conflict rate.

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage13.py --substage 13.5 --dry-run --seed 42`
- real: `python scripts/run_stage13.py --substage 13.5 --seed 42`

## 3) Validation gates & results
- gate_pass: `False`
- classification: `NO_EDGE`

## 4) Key metrics tables (trade_count, tpm, PF, expectancy, exp_lcb, maxDD, wf, mc)
- matrix: `docs/stage13_5_combined_matrix.csv`
- overlap: `docs/stage13_5_combined_overlap.csv`
- return correlation: `docs/stage13_5_combined_return_corr.csv`
- conflict: `docs/stage13_5_combined_conflict.csv`

## 5) Failures + reasons
- combined_gate_failed

## 6) Next actions
- Feed best combined mode into robustness sweeps (Stage-13.6).

## Summary
- best families: `price,volatility,flow`
- best composer_mode: `vote`
- best_exp_lcb: `0.000000`
