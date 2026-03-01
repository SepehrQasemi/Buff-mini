# Stage-13.7 Multi-Horizon Report

## 1) What changed
- Added rolling 3m/6m/12m non-overlapping horizon checks.

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage13.py --substage 13.7 --dry-run --seed 42`
- real: `python scripts/run_stage13.py --substage 13.7 --seed 42`

## 3) Validation gates & results
- horizon_consistency_score: `0.000000`
- classification: `NO_EDGE`

## 4) Key metrics tables (trade_count, tpm, PF, expectancy, exp_lcb, maxDD, wf, mc)
- windows table: `docs/stage13_7_multihorizon_table.csv`
- horizon scores: `docs/stage13_7_multihorizon_scores.csv`

## 5) Failures + reasons
- insufficient_horizon_consistency

## 6) Next actions
- Feed stable horizons into Stage-14 ML-lite calibration.
