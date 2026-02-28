# Stage-9 Impact Analysis

Conditional forward-return analysis from funding/OI regimes.

## BTC/USDT

- corr(funding_z_30, forward_return_24h): `-0.020045`
- corr(oi_z_30, forward_return_24h): `0.033907`

| condition | horizon | count_cond | count_base | median_cond | median_base | median_diff | ci_low | ci_high |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| funding_extreme_pos | 24h | 4398 | 23235 | 0.001230 | 0.000660 | 0.000570 | 0.000068 | 0.001108 |
| funding_extreme_pos | 72h | 4398 | 23187 | 0.004263 | 0.001650 | 0.002612 | 0.001617 | 0.003678 |
| funding_extreme_neg | 24h | 48 | 27585 | 0.003870 | 0.000746 | 0.003123 | -0.012610 | 0.009059 |
| funding_extreme_neg | 72h | 48 | 27537 | -0.001932 | 0.002144 | -0.004076 | -0.022125 | 0.006992 |
| crowd_long_risk | 24h | 0 | 27633 | 0.000000 | 0.000751 | 0.000000 | 0.000000 | 0.000000 |
| crowd_long_risk | 72h | 0 | 27585 | 0.000000 | 0.002137 | 0.000000 | 0.000000 | 0.000000 |
| crowd_short_risk | 24h | 0 | 27633 | 0.000000 | 0.000751 | 0.000000 | 0.000000 | 0.000000 |
| crowd_short_risk | 72h | 0 | 27585 | 0.000000 | 0.002137 | 0.000000 | 0.000000 | 0.000000 |

## ETH/USDT

- corr(funding_z_30, forward_return_24h): `-0.016867`
- corr(oi_z_30, forward_return_24h): `0.026062`

| condition | horizon | count_cond | count_base | median_cond | median_base | median_diff | ci_low | ci_high |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| funding_extreme_pos | 24h | 3534 | 24099 | 0.000696 | 0.000521 | 0.000175 | -0.000891 | 0.000976 |
| funding_extreme_pos | 72h | 3534 | 24051 | 0.006323 | 0.001168 | 0.005155 | 0.003313 | 0.007167 |
| funding_extreme_neg | 24h | 144 | 27489 | -0.009098 | 0.000567 | -0.009665 | -0.015939 | 0.000899 |
| funding_extreme_neg | 72h | 139 | 27446 | 0.000787 | 0.001658 | -0.000871 | -0.016735 | 0.005256 |
| crowd_long_risk | 24h | 0 | 27633 | 0.000000 | 0.000550 | 0.000000 | 0.000000 | 0.000000 |
| crowd_long_risk | 72h | 0 | 27585 | 0.000000 | 0.001656 | 0.000000 | 0.000000 | 0.000000 |
| crowd_short_risk | 24h | 18 | 27615 | -0.021731 | 0.000559 | -0.022290 | -0.042118 | 0.020032 |
| crowd_short_risk | 72h | 13 | 27572 | 0.069383 | 0.001649 | 0.067734 | 0.016392 | 0.076472 |

## Strongest Effects
- ETH/USDT | crowd_short_risk | 72h | median_diff=0.067734 | CI=[0.016392, 0.076472]
- ETH/USDT | crowd_short_risk | 24h | median_diff=-0.022290 | CI=[-0.042118, 0.020032]
- ETH/USDT | funding_extreme_neg | 24h | median_diff=-0.009665 | CI=[-0.015939, 0.000899]
