# Stage-3.1 Monte Carlo Cross-Check

- stage2_run_id: `20260227_015806_3cb775eb81a0_stage2`
- stage3_run_id: `20260227_043640_92029913e884_stage3_1_mc`
- initial_equity: `10000.0`

## Summary
- equal: status=PASS, trade_count_match=True, return_in_band=True, maxdd_within_p95=True, timestamp_inside_holdout=True
- vol: status=PASS, trade_count_match=True, return_in_band=True, maxdd_within_p95=True, timestamp_inside_holdout=True
- corr-min: status=PASS, trade_count_match=True, return_in_band=True, maxdd_within_p95=True, timestamp_inside_holdout=True

## Method Table
| method | status | trade_count_reconstructed | trade_count_stage3 | baseline_return_pct | mc_return_p05 | mc_return_p95 | baseline_max_dd | mc_maxdd_p95 | trade_count_match | return_in_band | maxdd_within_p95 | timestamp_inside_holdout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| equal | PASS | 218 | 218 | 0.0679 | 0.0051 | 0.1392 | 0.0219 | 0.0410 | True | True | True | True |
| vol | PASS | 218 | 218 | 0.0634 | 0.0078 | 0.1262 | 0.0172 | 0.0336 | True | True | True | True |
| corr-min | PASS | 175 | 175 | 0.0669 | -0.0093 | 0.1487 | 0.0290 | 0.0555 | True | True | True | True |

## Timestamp Ranges
- equal: trades `2025-01-29 12:00:00+00:00 .. 2026-01-27 09:00:00+00:00` vs holdout `2025-01-27T09:00:00+00:00..2026-01-27T09:00:00+00:00`
- vol: trades `2025-01-29 12:00:00+00:00 .. 2026-01-27 09:00:00+00:00` vs holdout `2025-01-27T09:00:00+00:00..2026-01-27T09:00:00+00:00`
- corr-min: trades `2025-01-29 12:00:00+00:00 .. 2026-01-27 09:00:00+00:00` vs holdout `2025-01-27T09:00:00+00:00..2026-01-27T09:00:00+00:00`
