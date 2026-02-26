# Stage-0.6 Cost Sensitivity Report

- Generated (UTC): 2026-02-26T10:31:56Z
- Python: 3.11.9
- Data mode: REAL (`data/raw/*.parquet`)
- Stage version: `stage0.6`
- Strategy count: 5 (S1-S5 fixed baselines)
- Gating default used: `vol+regime`

## Run IDs
- cost `1.0` -> `runs/20260226_103106_6fb682b4fd9e` (gating=`vol+regime`, window_months=`36`)
- cost `0.2` -> `runs/20260226_103123_b7885c9618cb` (gating=`vol+regime`, window_months=`36`)
- cost `0.1` -> `runs/20260226_103140_d86d3a51b822` (gating=`vol+regime`, window_months=`36`)

## Comparison Table

| cost | symbol | strategy | gating_mode | PF | expectancy | max_drawdown | trade_count | final_equity | date_range |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0.1 | BTC/USDT | Bollinger Mean Reversion | vol+regime | 0.694636 | -10.414903 | 0.831615 | 796 | 1709.737468 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.2 | BTC/USDT | Bollinger Mean Reversion | vol+regime | 0.587056 | -11.596029 | 0.924061 | 796 | 769.561169 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 1.0 | BTC/USDT | Bollinger Mean Reversion | vol+regime | 0.156507 | -12.561232 | 0.999875 | 796 | 1.259394 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.1 | BTC/USDT | Donchian Breakout | vol+regime | 0.824444 | -12.111897 | 0.576693 | 418 | 4937.227095 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.2 | BTC/USDT | Donchian Breakout | vol+regime | 0.738897 | -16.156559 | 0.705481 | 418 | 3246.558495 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 1.0 | BTC/USDT | Donchian Breakout | vol+regime | 0.307047 | -23.656049 | 0.989484 | 418 | 111.771510 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.1 | BTC/USDT | RSI Mean Reversion | vol+regime | 0.768617 | -13.147904 | 0.483704 | 361 | 5253.606772 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.2 | BTC/USDT | RSI Mean Reversion | vol+regime | 0.665243 | -17.566694 | 0.639940 | 361 | 3658.423346 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 1.0 | BTC/USDT | RSI Mean Reversion | vol+regime | 0.209376 | -27.147771 | 0.980222 | 361 | 199.654579 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.1 | BTC/USDT | Range Breakout w/ EMA Trend Filter | vol+regime | 0.776142 | -15.294982 | 0.595583 | 347 | 4692.641232 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.2 | BTC/USDT | Range Breakout w/ EMA Trend Filter | vol+regime | 0.696134 | -19.272157 | 0.705428 | 347 | 3312.561424 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 1.0 | BTC/USDT | Range Breakout w/ EMA Trend Filter | vol+regime | 0.306628 | -28.237267 | 0.981027 | 347 | 201.668522 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.1 | BTC/USDT | Trend Pullback | vol+regime | 0.754157 | -15.938752 | 0.171012 | 78 | 8756.777371 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.2 | BTC/USDT | Trend Pullback | vol+regime | 0.642642 | -24.376946 | 0.227256 | 78 | 8098.598185 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 1.0 | BTC/USDT | Trend Pullback | vol+regime | 0.181430 | -72.792528 | 0.576994 | 78 | 4322.182813 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.1 | ETH/USDT | Bollinger Mean Reversion | vol+regime | 0.788300 | -10.173511 | 0.868371 | 849 | 1362.689498 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.2 | ETH/USDT | Bollinger Mean Reversion | vol+regime | 0.702572 | -11.093555 | 0.943655 | 849 | 581.572127 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 1.0 | ETH/USDT | Bollinger Mean Reversion | vol+regime | 0.247184 | -11.777832 | 0.999939 | 849 | 0.620616 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.1 | ETH/USDT | Donchian Breakout | vol+regime | 0.900688 | -7.896044 | 0.464152 | 401 | 6833.686509 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.2 | ETH/USDT | Donchian Breakout | vol+regime | 0.803927 | -13.534618 | 0.595584 | 401 | 4572.617991 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 1.0 | ETH/USDT | Donchian Breakout | vol+regime | 0.282008 | -24.486045 | 0.982111 | 401 | 181.095912 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.1 | ETH/USDT | RSI Mean Reversion | vol+regime | 0.894832 | -7.585399 | 0.480006 | 387 | 7064.450740 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.2 | ETH/USDT | RSI Mean Reversion | vol+regime | 0.788916 | -13.445795 | 0.601525 | 387 | 4796.477443 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 1.0 | ETH/USDT | RSI Mean Reversion | vol+regime | 0.256768 | -25.287883 | 0.978666 | 387 | 213.589349 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.1 | ETH/USDT | Range Breakout w/ EMA Trend Filter | vol+regime | 0.995492 | -0.448929 | 0.387862 | 335 | 9849.608917 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.2 | ETH/USDT | Range Breakout w/ EMA Trend Filter | vol+regime | 0.899736 | -8.826651 | 0.440001 | 335 | 7043.071845 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 1.0 | ETH/USDT | Range Breakout w/ EMA Trend Filter | vol+regime | 0.357204 | -28.431108 | 0.953059 | 335 | 475.578889 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.1 | ETH/USDT | Trend Pullback | vol+regime | 0.758620 | -21.937777 | 0.236545 | 75 | 8354.666689 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 0.2 | ETH/USDT | Trend Pullback | vol+regime | 0.676029 | -30.013645 | 0.278556 | 75 | 7748.976592 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |
| 1.0 | ETH/USDT | Trend Pullback | vol+regime | 0.260412 | -76.902468 | 0.598242 | 75 | 4232.314866 | 2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00 |

## Conclusion
- No Stage-0.6 strategy meets `PF > 1` and `expectancy > 0` at lower costs (`0.2` or `0.1`).
- Under current assumptions, expanded feasibility remains weak and Phase-2 alone is unlikely to unlock edge.

## Best Observed Row
- cost=0.1, symbol=ETH/USDT, strategy=Range Breakout w/ EMA Trend Filter, gating=vol+regime, PF=0.995492, expectancy=-0.448929, date_range=2023-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00
