# Stage-0 Cost Sensitivity Report

- Generated (UTC): 2026-02-26T09:48:58Z
- Python: 3.11.9
- Mode: REAL data (no dry-run)
- Symbols/timeframe/date range/seed were held constant; only `costs.round_trip_cost_pct` changed.

## Run IDs
- cost `1.0` -> `runs/20260226_094826_85a7f2023b3a`
- cost `0.2` -> `runs/20260226_094837_5bf3d8b4c424`
- cost `0.1` -> `runs/20260226_094848_30dbdc5f9765`

## Comparison Table

| cost | symbol | strategy | PF | expectancy | max_drawdown | trade_count | final_equity |
|---:|---|---|---:|---:|---:|---:|---:|
| 0.1 | BTC/USDT | Donchian Breakout | 0.745082 | -7.411359 | 0.951132 | 1275 | 550.517017 |
| 0.2 | BTC/USDT | Donchian Breakout | 0.656110 | -7.722946 | 0.985389 | 1275 | 153.243753 |
| 1.0 | BTC/USDT | Donchian Breakout | 0.235081 | -7.843133 | 0.999999 | 1275 | 0.005274 |
| 0.1 | BTC/USDT | RSI Mean Reversion | 0.600148 | -6.138583 | 0.975037 | 1583 | 282.623216 |
| 0.2 | BTC/USDT | RSI Mean Reversion | 0.475990 | -6.280630 | 0.994772 | 1583 | 57.763081 |
| 1.0 | BTC/USDT | RSI Mean Reversion | 0.097394 | -6.317119 | 1.000000 | 1583 | 0.000166 |
| 0.1 | BTC/USDT | Trend Pullback | 0.737150 | -12.817295 | 0.431284 | 262 | 6641.868746 |
| 0.2 | BTC/USDT | Trend Pullback | 0.611441 | -18.672264 | 0.554051 | 262 | 5107.866919 |
| 1.0 | BTC/USDT | Trend Pullback | 0.139498 | -35.805324 | 0.940337 | 262 | 619.004986 |
| 0.1 | ETH/USDT | Donchian Breakout | 0.734131 | -7.395178 | 0.942920 | 1272 | 593.333608 |
| 0.2 | ETH/USDT | Donchian Breakout | 0.630609 | -7.731407 | 0.983896 | 1272 | 165.650670 |
| 1.0 | ETH/USDT | Donchian Breakout | 0.211270 | -7.861631 | 0.999999 | 1272 | 0.005837 |
| 0.1 | ETH/USDT | RSI Mean Reversion | 0.773435 | -6.229028 | 0.980519 | 1572 | 207.968116 |
| 0.2 | ETH/USDT | RSI Mean Reversion | 0.689824 | -6.333984 | 0.995778 | 1572 | 42.976978 |
| 1.0 | ETH/USDT | RSI Mean Reversion | 0.264651 | -6.361323 | 1.000000 | 1572 | 0.000135 |
| 0.1 | ETH/USDT | Trend Pullback | 0.826558 | -10.173820 | 0.522607 | 270 | 7253.068469 |
| 0.2 | ETH/USDT | Trend Pullback | 0.703797 | -16.540645 | 0.603336 | 270 | 5534.025959 |
| 1.0 | ETH/USDT | Trend Pullback | 0.179441 | -34.705891 | 0.941594 | 270 | 629.409557 |

## Conclusion
- No baseline met `PF > 1` and `expectancy > 0` at `0.2` or `0.1`.
- Phase-2 discovery is unlikely to help without changing assumptions.

## Sanity Check
- BTC/USDT / Donchian Breakout: equity(0.1) >= equity(0.2) [550.5170 vs 153.2438]
- BTC/USDT / RSI Mean Reversion: equity(0.1) >= equity(0.2) [282.6232 vs 57.7631]
- BTC/USDT / Trend Pullback: equity(0.1) >= equity(0.2) [6641.8687 vs 5107.8669]
- ETH/USDT / Donchian Breakout: equity(0.1) >= equity(0.2) [593.3336 vs 165.6507]
- ETH/USDT / RSI Mean Reversion: equity(0.1) >= equity(0.2) [207.9681 vs 42.9770]
- ETH/USDT / Trend Pullback: equity(0.1) >= equity(0.2) [7253.0685 vs 5534.0260]
