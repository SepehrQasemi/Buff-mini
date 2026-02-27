# Stage-3.3 Leverage Selector Summary

- run_id: `20260227_113410_aca9cf2325a2_stage3_3_selector`
- commit hash: `N/A`
- chosen overall: `equal` @ `5.0x`
- binding constraint (chosen): `min_return_p05`
- constraints: `max_p_ruin=0.01, max_dd_p95=0.25, min_return_p05=0.0`

## Leverage Snapshot (1,2,3,5,10)
| method | leverage | expected_log_growth | return_p05 | maxDD_p95 | P(ruin) | feasible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| equal | 1 | 0.067897 | 0.005088 | 0.040975 | 0.000000 | True |
| equal | 2 | 0.130264 | 0.009585 | 0.079445 | 0.000000 | True |
| equal | 3 | 0.187943 | 0.016325 | 0.117243 | 0.000000 | True |
| equal | 5 | 0.291637 | 0.022029 | 0.186519 | 0.000000 | True |
| equal | 10 | 0.502900 | 0.042815 | 0.345931 | 0.005100 | False |
| vol | 1 | 0.063783 | 0.007765 | 0.033835 | 0.000000 | True |
| vol | 2 | 0.122260 | 0.015443 | 0.066235 | 0.000000 | True |
| vol | 3 | 0.176004 | 0.022684 | 0.097379 | 0.000000 | True |
| vol | 5 | 0.276577 | 0.037223 | 0.156152 | 0.000000 | True |
| vol | 10 | 0.484203 | 0.081438 | 0.285207 | 0.001450 | False |

## Interpretation
Both `equal` and `vol` remain feasible through 5x under the configured risk limits, then fail at higher leverage mainly due to drawdown and ruin constraints. The selector chooses `equal` at `5.0x` because it has the strongest expected log-growth inside the feasible set.
