# Stage-0.5 Cost Sensitivity Report

- Generated (UTC): 2026-02-26T10:06:28Z
- Python: 3.11.9
- Mode: REAL data (no dry-run)
- Stage-0.5 filters enabled:
  1) Volatility gate: `ATR(14) > SMA(ATR(14), 50)`
  2) Optional regime gate per strategy: longs only `EMA50>EMA200`, shorts only `EMA50<EMA200`
- Symbols/timeframe/date range/seed were held constant; only `costs.round_trip_cost_pct` changed.

## Run IDs
- cost `1.0` -> `runs/20260226_100556_85a7f2023b3a`
- cost `0.2` -> `runs/20260226_100607_5bf3d8b4c424`
- cost `0.1` -> `runs/20260226_100618_30dbdc5f9765`

## Comparison Table

| cost | symbol | strategy | PF | expectancy | max_drawdown | trade_count | final_equity |
|---:|---|---|---:|---:|---:|---:|---:|
| 0.1 | BTC/USDT | Donchian Breakout | 0.846335 | -11.039979 | 0.576693 | 437 | 5175.529158 |
| 0.2 | BTC/USDT | Donchian Breakout | 0.760841 | -15.242035 | 0.705481 | 437 | 3339.230511 |
| 1.0 | BTC/USDT | Donchian Breakout | 0.323700 | -22.657463 | 0.990131 | 437 | 98.688787 |
| 0.1 | BTC/USDT | RSI Mean Reversion | 0.779466 | -12.648526 | 0.505321 | 378 | 5218.857078 |
| 0.2 | BTC/USDT | RSI Mean Reversion | 0.676883 | -17.003114 | 0.660169 | 378 | 3572.823097 |
| 1.0 | BTC/USDT | RSI Mean Reversion | 0.222086 | -26.005211 | 0.983371 | 378 | 170.030145 |
| 0.1 | BTC/USDT | Trend Pullback | 0.752735 | -16.029067 | 0.185900 | 83 | 8669.587422 |
| 0.2 | BTC/USDT | Trend Pullback | 0.641728 | -24.363769 | 0.244181 | 83 | 7977.807140 |
| 1.0 | BTC/USDT | Trend Pullback | 0.179311 | -71.212681 | 0.599781 | 83 | 4089.347456 |
| 0.1 | ETH/USDT | Donchian Breakout | 0.872840 | -9.259104 | 0.509146 | 418 | 6129.694664 |
| 0.2 | ETH/USDT | Donchian Breakout | 0.773479 | -14.277121 | 0.636163 | 418 | 4032.163336 |
| 1.0 | ETH/USDT | Donchian Breakout | 0.252539 | -23.590363 | 0.986077 | 418 | 139.228364 |
| 0.1 | ETH/USDT | RSI Mean Reversion | 0.890637 | -7.766359 | 0.534385 | 412 | 6800.260233 |
| 0.2 | ETH/USDT | RSI Mean Reversion | 0.785691 | -13.343273 | 0.648969 | 412 | 4502.571608 |
| 1.0 | ETH/USDT | RSI Mean Reversion | 0.273063 | -23.874148 | 0.983886 | 412 | 163.851198 |
| 0.1 | ETH/USDT | Trend Pullback | 0.833789 | -15.115702 | 0.236545 | 79 | 8805.859528 |
| 0.2 | ETH/USDT | Trend Pullback | 0.744329 | -23.607157 | 0.278556 | 79 | 8135.034575 |
| 1.0 | ETH/USDT | Trend Pullback | 0.284660 | -72.109044 | 0.599582 | 79 | 4303.385552 |

## Conclusion
- No baseline meets `PF > 1` and `expectancy > 0` at `0.2` or `0.1` under Stage-0.5 filters.
- Additional discovery alone is unlikely to help without changing assumptions or execution constraints.
