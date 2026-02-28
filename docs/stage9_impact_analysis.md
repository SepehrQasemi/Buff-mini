# Stage-9 Impact Analysis

Conditional forward-return analysis from funding/OI regimes.

## BTC/USDT

- corr(funding_z_30, forward_return_24h): `-0.020045`
- corr(oi_z_30, forward_return_24h): `-0.228708`

| condition | horizon | count_cond | count_base | median_cond | median_base | median_diff | ci_low | ci_high |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| funding_extreme_pos | 24h | 4398 | 23235 | 0.001230 | 0.000660 | 0.000570 | 0.000068 | 0.001108 |
| funding_extreme_pos | 72h | 4398 | 23187 | 0.004263 | 0.001650 | 0.002612 | 0.001617 | 0.003678 |
| funding_extreme_neg | 24h | 48 | 27585 | 0.003870 | 0.000746 | 0.003123 | -0.012610 | 0.009059 |
| funding_extreme_neg | 72h | 48 | 27537 | -0.001932 | 0.002144 | -0.004076 | -0.022125 | 0.006992 |
| crowd_long_risk | 24h | 0 | 27633 | 0.000000 | 0.000751 | 0.000000 | 0.000000 | 0.000000 |
| crowd_long_risk | 72h | 0 | 27585 | 0.000000 | 0.002137 | 0.000000 | 0.000000 | 0.000000 |
| crowd_short_risk | 24h | 1 | 27632 | 0.008602 | 0.000750 | 0.007851 | 0.007652 | 0.008042 |
| crowd_short_risk | 72h | 1 | 27584 | 0.002206 | 0.002137 | 0.000069 | -0.000379 | 0.000477 |

## ETH/USDT

- corr(funding_z_30, forward_return_24h): `-0.016867`
- corr(oi_z_30, forward_return_24h): `-0.103065`

| condition | horizon | count_cond | count_base | median_cond | median_base | median_diff | ci_low | ci_high |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| funding_extreme_pos | 24h | 3534 | 24099 | 0.000696 | 0.000521 | 0.000175 | -0.000891 | 0.000976 |
| funding_extreme_pos | 72h | 3534 | 24051 | 0.006323 | 0.001168 | 0.005155 | 0.003313 | 0.007167 |
| funding_extreme_neg | 24h | 144 | 27489 | -0.009098 | 0.000567 | -0.009665 | -0.015939 | 0.000899 |
| funding_extreme_neg | 72h | 139 | 27446 | 0.000787 | 0.001658 | -0.000871 | -0.016735 | 0.005256 |
| crowd_long_risk | 24h | 7 | 27626 | 0.014300 | 0.000550 | 0.013750 | -0.000513 | 0.016256 |
| crowd_long_risk | 72h | 7 | 27578 | -0.007617 | 0.001656 | -0.009274 | -0.016634 | 0.024567 |
| crowd_short_risk | 24h | 34 | 27599 | -0.027689 | 0.000570 | -0.028259 | -0.046314 | -0.020978 |
| crowd_short_risk | 72h | 29 | 27556 | -0.154323 | 0.001658 | -0.155981 | -0.175131 | 0.016671 |

## Strongest Effects
- ETH/USDT | crowd_short_risk | 72h | median_diff=-0.155981 | CI=[-0.175131, 0.016671]
- ETH/USDT | crowd_short_risk | 24h | median_diff=-0.028259 | CI=[-0.046314, -0.020978]
- ETH/USDT | crowd_long_risk | 24h | median_diff=0.013750 | CI=[-0.000513, 0.016256]
