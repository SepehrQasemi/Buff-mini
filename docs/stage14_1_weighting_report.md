# Stage-14.1 ML-lite Weighting Report

## 1) What changed
- Trained deterministic regularized linear model on family scores.

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage13.py --substage 14.1 --dry-run --seed 42`
- real: `python scripts/run_stage13.py --substage 14.1 --seed 42`

## 3) Validation gates & results
- best model: `logreg_l2`
- holdout_exp_lcb: `0.000000`
- forward_exp_lcb: `0.000000`
- drift_ok: `True`

## 4) Key metrics tables (trade_count, tpm, PF, expectancy, exp_lcb, maxDD, wf, mc)
- forward trade_count: `0.00`
- forward tpm: `0.00`

## 5) Failures + reasons
- none

## 6) Next actions
- Calibrate thresholds per regime (Stage-14.2).
