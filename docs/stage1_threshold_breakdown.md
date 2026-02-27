# Stage-1 Threshold Breakdown

- run_id: `20260226_234139_b78b63b21ba0_stage1`
- source_csv: `runs\20260226_234139_b78b63b21ba0_stage1\near_miss_candidates.csv`
- joined_json: `docs\stage1_real_data_report.json`
- near_miss_count: `29`
- thresholds: `{'min_exp_lcb_holdout': 0, 'min_effective_edge': 0, 'min_trades_per_month_holdout': 5, 'min_pf_adj_holdout': 1.1, 'max_drawdown_holdout': 0.15, 'min_exposure_ratio': 0.02}`

## Failure Counts

| threshold | failed_near_miss_count |
| --- | ---: |
| exp_lcb_holdout <= min_exp_lcb_holdout | 2 |
| effective_edge <= min_effective_edge | 2 |
| tpm_holdout < min_trades_per_month_holdout | 28 |
| pf_adj_holdout < min_pf_adj_holdout | 0 |
| dd_holdout > max_drawdown_holdout | 0 |
| exposure_ratio < min_exposure_ratio | 21 |

- Biggest bottleneck: `tpm_holdout < min_trades_per_month_holdout` (28 failures)

## Distance Method

- `exp_lcb_holdout` and `effective_edge` use absolute shortfall to zero because their Tier A thresholds are zero.
- `tpm_holdout`, `pf_adj_holdout`, `dd_holdout`, and `exposure_ratio` use normalized gap/excess versus the configured threshold.
- `total_distance` is the sum of all six per-threshold distances. Smaller is closer to Tier A.
- `dd_holdout` is joined from `docs/stage1_real_data_report.json` because `near_miss_candidates.csv` does not currently persist that field.

## Closest Near Misses

| rank | candidate_id | strategy | gating | exit | holdout_m | failed_thresholds | total_distance | exp_lcb | edge | tpm | pf_adj | dd_holdout | exposure |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | cand_001264_848f658f | TrendPullback | vol | breakeven_1r | 3 | tpm | 0.4130 | 44.3964 | 130.2938 | 2.9348 | 2.3729 | 0.0032 | 0.0326 |
| 2 | cand_000637_42edc0b1 | TrendPullback | vol | fixed_atr | 3 | tpm | 0.4783 | 58.7705 | 153.3145 | 2.6087 | 2.2414 | 0.0102 | 0.0251 |
| 3 | cand_001139_d55f8520 | TrendPullback | vol | fixed_atr | 3 | tpm | 0.4783 | 39.3389 | 102.6232 | 2.6087 | 2.2414 | 0.0170 | 0.0204 |
| 4 | cand_001012_dbec7b03 | TrendPullback | none | fixed_atr | 9 | tpm | 0.5000 | 14.6575 | 36.6439 | 2.5000 | 1.4985 | 0.0177 | 0.0205 |
| 5 | cand_001147_0131018a | TrendPullback | vol | fixed_atr | 12 | tpm | 0.5562 | 9.1897 | 20.3936 | 2.2192 | 1.2357 | 0.0554 | 0.0237 |
| 6 | cand_001603_418d4b75 | TrendPullback | vol+regime | breakeven_1r | 12 | tpm | 0.5726 | 12.2663 | 26.2129 | 2.1370 | 1.2845 | 0.0466 | 0.0218 |
| 7 | cand_000839_134b1afb | TrendPullback | vol+regime | fixed_atr | 12 | tpm | 0.6219 | 4.9367 | 9.3324 | 1.8904 | 1.2213 | 0.0542 | 0.0227 |
| 8 | cand_001674_4d2aff0a | TrendPullback | vol | breakeven_1r | 12 | tpm, exposure | 1.1090 | 19.2792 | 38.0302 | 1.9726 | 1.6196 | 0.0198 | 0.0099 |
| 9 | cand_001606_0bb0538b | TrendPullback | vol | fixed_atr | 12 | tpm, exposure | 1.2176 | 42.3147 | 55.6467 | 1.3151 | 1.4816 | 0.0309 | 0.0104 |
| 10 | cand_000574_79a0305f | TrendPullback | vol | fixed_atr | 12 | tpm, exposure | 1.2376 | 43.5536 | 57.2759 | 1.3151 | 1.7578 | 0.0172 | 0.0100 |

## Notes

- Counts are non-exclusive: one near-miss candidate can fail multiple Tier A thresholds.
- This report is derived from the latest run artifacts only; it does not change Stage-1 execution behavior.
