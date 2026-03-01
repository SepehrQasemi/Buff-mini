# Stage-13.6 Robustness Report

## 1) What changed
- Added cost sensitivity, execution drag, and deterministic regime-stress proxy sweeps.

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage13.py --substage 13.6 --dry-run --seed 42`
- real: `python scripts/run_stage13.py --substage 13.6 --seed 42`

## 3) Validation gates & results
- classification: `NO_EDGE`
- best robust_score: `-1.000000`

## 4) Key metrics tables (trade_count, tpm, PF, expectancy, exp_lcb, maxDD, wf, mc)
- table: `docs/stage13_6_robustness_table.csv`

## 5) Failures + reasons
- cost_or_drag_collapse

## 6) Next actions
- Use best robust config for multi-horizon validation.
