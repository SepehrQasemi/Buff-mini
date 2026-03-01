# Stage-13.2 Price Family Report

## 1) What changed
- Ran bounded sweep for `price` family with deterministic configs.

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage13.py --substage 13.2 --family price --dry-run --seed 42`
- real: `python scripts/run_stage13.py --substage 13.2 --family price --seed 42`

## 3) Validation gates & results
- zero_trade_pct_min: `0.000000`
- trade_count_ratio_vs_baseline: `0.151515`
- walkforward_executed_true_pct_max: `0.000000`
- mc_trigger_rate_max: `100.000000`

## 4) Key metrics tables (trade_count, tpm, PF, expectancy, exp_lcb, maxDD, wf, mc)
- sweep table: `docs/stage13_2_price_family_table.csv`

## 5) Failures + reasons
- trade_count_ratio_gate_failed

## 6) Next actions
- Use best family configs in combined composer matrix (Stage-13.5).

## Summary
- classification: `NO_EDGE`
- best_exp_lcb: `277.709248`
- best_trade_count: `2.50`
