# Stage-27.9 Rolling Discovery Summary

## Config
- run_id: `20260304_114018_3fbd26976466_stage27_9_roll`
- seed: `42`
- dry_run: `True`
- symbols: `['BTC/USDT']`
- timeframes: `['1h']`
- windows: `[3, 6]`
- step_months: `1`

## Window Coverage
- 3m: generated=1, evaluated=1
- 6m: generated=0, evaluated=0

## Metrics
- rows: `17`
- positive_exp_lcb_rows: `0`
- runtime_seconds: `2.982253`

## Top Contextual Rows
| symbol | timeframe | window_months | context | rulelet | trade_count | exp | exp_lcb | pf |
|---|---:|---:|---|---|---:|---:|---:|---:|
| BTC/USDT | 1h | 3 | TREND | BreakoutRetest | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 1h | 3 | VOL_EXPANSION | BreakoutRetest | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 1h | 3 | CHOP | ChopFilterGate | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 1h | 3 | CHOP | TrendFlip | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 1h | 3 | TREND | TrendFlip | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 1h | 3 | TREND | MomentumBurst | 63 | -10.656472 | -53.278978 | 0.997885 |
| BTC/USDT | 1h | 3 | VOL_EXPANSION | VolExpansionContinuation | 89 | -28.895295 | -53.352542 | 0.600133 |
| BTC/USDT | 1h | 3 | TREND | StructureBreak | 36 | -16.866518 | -54.538190 | 0.565867 |
| BTC/USDT | 1h | 3 | TREND | TrendPullback | 68 | -17.897545 | -61.156942 | 0.916123 |
| BTC/USDT | 1h | 3 | VOL_COMPRESSION | VolCompressionBreakout | 20 | -45.174346 | -72.215018 | 0.376495 |
| BTC/USDT | 1h | 3 | RANGE | RangeFade | 64 | -42.958517 | -75.423614 | 0.574344 |
| BTC/USDT | 1h | 3 | RANGE | BollingerSnapBack | 60 | -42.459295 | -78.259521 | 0.557388 |
| BTC/USDT | 1h | 3 | RANGE | MeanRevertAfterSpike | 59 | -56.749311 | -89.045659 | 0.500494 |
| BTC/USDT | 1h | 3 | VOL_EXPANSION | FailedBreakReversal | 39 | -61.893759 | -101.052337 | 0.374110 |
| BTC/USDT | 1h | 3 | RANGE | FailedBreakReversal | 30 | -85.957179 | -127.741963 | 0.296780 |
