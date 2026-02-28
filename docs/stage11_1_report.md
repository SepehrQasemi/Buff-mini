# Stage-11.1 Effectiveness Report

## 1) What changed vs prior Stage-11
- Stage-11.1 enables explicit presets for bias/confirm/bias+confirm and enforces no-op detection.
- Runs fail with `NO-OP BUG` when hooks are enabled but do not cause measurable effects.

## 2) Presets
- `stage11_bias`: 4h context bias only, multiplier clamp 0.85..1.15.
- `stage11_confirm`: 15m confirm only, delay up to 3 base bars, deterministic thresholding.
- `stage11_bias_confirm`: both hooks enabled.

## 3) Mode Results
| dataset | window_months | mode | verdict | trade_count | PF | expectancy | exp_lcb | maxDD | tpm | trade_count_delta_pct | guard_pass | wf_class | wf_usable |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| synthetic | 3 | baseline | NEUTRAL | 417.0 | 0.423849 | -27.408537 | -27.436962 | 0.620832 | 417.000 | 0.000% | True | N/A | 0 |
| synthetic | 3 | bias | NEUTRAL | 417.0 | 0.423358 | -27.269666 | -27.306611 | 0.617484 | 417.000 | 0.000% | True | N/A | 0 |
| synthetic | 3 | confirm | REGRESSION | 299.0 | 0.455711 | -29.226112 | -29.324113 | 0.481283 | 299.000 | -28.297% | False | N/A | 0 |
| synthetic | 3 | bias_confirm | REGRESSION | 299.0 | 0.455011 | -29.080870 | -29.187325 | 0.478441 | 299.000 | -28.297% | False | N/A | 0 |
| real | 3 | baseline | NEUTRAL | 36.0 | 0.897344 | -5.326175 | -6.268783 | 0.056966 | 36.000 | 0.000% | True | INSUFFICIENT_DATA | 0 |
| real | 3 | bias | NEUTRAL | 36.0 | 0.892156 | -5.716081 | -6.583272 | 0.058804 | 36.000 | 0.000% | True | INSUFFICIENT_DATA | 0 |
| real | 3 | confirm | NEUTRAL | 36.0 | 0.897344 | -5.326175 | -6.268783 | 0.056966 | 36.000 | 0.000% | True | INSUFFICIENT_DATA | 0 |
| real | 3 | bias_confirm | NEUTRAL | 36.0 | 0.892156 | -5.716081 | -6.583272 | 0.058804 | 36.000 | 0.000% | True | INSUFFICIENT_DATA | 0 |
| real | 12 | baseline | NEUTRAL | 48.0 | 1.707612 | 28.223445 | 5.905885 | 0.084667 | 48.000 | 0.000% | True | INSUFFICIENT_DATA | 0 |
| real | 12 | bias | NEUTRAL | 48.0 | 1.715637 | 28.069964 | 6.326618 | 0.082967 | 48.000 | 0.000% | True | INSUFFICIENT_DATA | 0 |
| real | 12 | confirm | NEUTRAL | 48.0 | 1.707612 | 28.223445 | 5.905885 | 0.084667 | 48.000 | 0.000% | True | INSUFFICIENT_DATA | 0 |
| real | 12 | bias_confirm | NEUTRAL | 48.0 | 1.715637 | 28.069964 | 6.326618 | 0.082967 | 48.000 | 0.000% | True | INSUFFICIENT_DATA | 0 |

## 4) Effectiveness Proof
- `synthetic` bias sizing stats: mean=1.016882, p05=0.964639, p95=1.096689, pct_not_1.0=0.707
- `synthetic` confirm confirm stats: seen=1639, confirmed=944, skipped=695, confirm_rate=0.576, median_delay=0.00
- `synthetic` bias_confirm confirm stats: seen=1639, confirmed=944, skipped=695, confirm_rate=0.576, median_delay=0.00
- `real` bias sizing stats: mean=1.011252, p05=0.952436, p95=1.079735, pct_not_1.0=0.706
- `real` confirm confirm stats: seen=1711, confirmed=1059, skipped=652, confirm_rate=0.619, median_delay=0.00
- `real` bias_confirm confirm stats: seen=1711, confirmed=1059, skipped=652, confirm_rate=0.619, median_delay=0.00
- `real` bias sizing stats: mean=1.007577, p05=0.955813, p95=1.073640, pct_not_1.0=0.930
- `real` confirm confirm stats: seen=6998, confirmed=4612, skipped=2386, confirm_rate=0.659, median_delay=0.00
- `real` bias_confirm confirm stats: seen=6998, confirmed=4612, skipped=2386, confirm_rate=0.659, median_delay=0.00

## 5) Best Mode
- Synthetic best mode: `bias`
- Real best mode: `bias`

## 6) Notes
- none
