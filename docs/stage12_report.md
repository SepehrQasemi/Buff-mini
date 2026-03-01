# Stage-12 Full Price-Family Robustness Sweep

- run_id: `20260301_060514_8bb2fcf2c8c0_stage12`
- symbols: `BTC/USDT, ETH/USDT`
- timeframes: `15m, 30m, 1h, 2h, 4h, 1d`
- total_combinations: `1188`
- valid_combinations: `0`
- runtime_seconds: `50.431`
- verdict: `NO ROBUST EDGE`

## Runtime Breakdown by Timeframe
| timeframe | runtime_seconds |
| --- | ---: |
| 15m | 17.1663 |
| 1d | 2.5342 |
| 1h | 9.2598 |
| 2h | 6.4689 |
| 30m | 11.0832 |
| 4h | 2.6721 |

## Top 10 Robust Combinations
| symbol | timeframe | strategy | exit_type | cost_level | exp_lcb | PF | expectancy | robust_score | cost_sensitivity | stability | MC_p_ruin | MC_p_return_negative |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| BTC/USDT | 1h | RSI Mean Reversion | time_based | high | 1064.693271 | 0.000000 | 1064.693271 | 877.753470 | 61.979933 | INVALID | nan | nan |
| BTC/USDT | 1h | RSI Mean Reversion | time_based | low | 940.733404 | 0.000000 | 940.733404 | 877.753470 | 61.979933 | INVALID | nan | nan |
| BTC/USDT | 1h | RSI Mean Reversion | time_based | realistic | 940.733404 | 0.000000 | 940.733404 | 877.753470 | 61.979933 | INVALID | nan | nan |
| BTC/USDT | 1h | ATR Distance Revert | time_based | high | 1045.445099 | 0.000000 | 1045.445099 | 858.828614 | 61.872162 | INVALID | nan | nan |
| BTC/USDT | 1h | Bollinger Mean Reversion | time_based | high | 1045.445099 | 0.000000 | 1045.445099 | 858.828614 | 61.872162 | INVALID | nan | nan |
| BTC/USDT | 1h | ATR Distance Revert | time_based | low | 921.700775 | 0.000000 | 921.700775 | 858.828614 | 61.872162 | INVALID | nan | nan |
| BTC/USDT | 1h | ATR Distance Revert | time_based | realistic | 921.700775 | 0.000000 | 921.700775 | 858.828614 | 61.872162 | INVALID | nan | nan |
| BTC/USDT | 1h | Bollinger Mean Reversion | time_based | low | 921.700775 | 0.000000 | 921.700775 | 858.828614 | 61.872162 | INVALID | nan | nan |
| BTC/USDT | 1h | Bollinger Mean Reversion | time_based | realistic | 921.700775 | 0.000000 | 921.700775 | 858.828614 | 61.872162 | INVALID | nan | nan |
| ETH/USDT | 15m | Range Fade | time_based | high | 664.495842 | 0.000000 | 664.495842 | 589.202124 | 24.764573 | INVALID | nan | nan |

## Per-Timeframe Summary
| timeframe | combos | valid | exp_lcb_median | best_robust_score |
| --- | ---: | ---: | ---: | ---: |
| 15m | 198 | 0 | -84.815732 | 589.202124 |
| 1d | 198 | 0 | 0.000000 | -1.000000 |
| 1h | 198 | 0 | 0.000000 | 877.753470 |
| 2h | 198 | 0 | 0.000000 | -1.000000 |
| 30m | 198 | 0 | 0.000000 | 357.192138 |
| 4h | 198 | 0 | 0.000000 | -1.000000 |

## Per-Symbol Summary
| symbol | combos | valid | exp_lcb_median | best_robust_score |
| --- | ---: | ---: | ---: | ---: |
| BTC/USDT | 594 | 0 | 0.000000 | 877.753470 |
| ETH/USDT | 594 | 0 | 0.000000 | 589.202124 |

## Stability Heatmap Table
| symbol | timeframe | STABLE | UNSTABLE | ZERO_TRADE | INVALID | INSUFFICIENT_DATA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BTC/USDT | 15m | 0 | 0 | 27 | 72 | 0 |
| BTC/USDT | 1d | 0 | 0 | 99 | 0 | 0 |
| BTC/USDT | 1h | 0 | 0 | 54 | 45 | 0 |
| BTC/USDT | 2h | 0 | 0 | 99 | 0 | 0 |
| BTC/USDT | 30m | 0 | 0 | 45 | 54 | 0 |
| BTC/USDT | 4h | 0 | 0 | 99 | 0 | 0 |
| ETH/USDT | 15m | 0 | 0 | 18 | 81 | 0 |
| ETH/USDT | 1d | 0 | 0 | 99 | 0 | 0 |
| ETH/USDT | 1h | 0 | 0 | 54 | 45 | 0 |
| ETH/USDT | 2h | 0 | 0 | 99 | 0 | 0 |
| ETH/USDT | 30m | 0 | 0 | 63 | 36 | 0 |
| ETH/USDT | 4h | 0 | 0 | 99 | 0 | 0 |

## Cost Sensitivity Chart Data
| symbol | timeframe | strategy | exit_type | cost_level | exp_lcb |
| --- | --- | --- | --- | --- | ---: |
| BTC/USDT | 15m | ATR Distance Revert | fixed_atr | high | -179.913175 |
| BTC/USDT | 15m | ATR Distance Revert | fixed_atr | low | -160.403040 |
| BTC/USDT | 15m | ATR Distance Revert | fixed_atr | realistic | -160.403040 |
| BTC/USDT | 15m | ATR Distance Revert | structure_trailing | high | -150.947350 |
| BTC/USDT | 15m | ATR Distance Revert | structure_trailing | low | -87.362727 |
| BTC/USDT | 15m | ATR Distance Revert | structure_trailing | realistic | -87.362727 |
| BTC/USDT | 15m | ATR Distance Revert | time_based | high | -275.494469 |
| BTC/USDT | 15m | ATR Distance Revert | time_based | low | -222.632101 |
| BTC/USDT | 15m | ATR Distance Revert | time_based | realistic | -222.632101 |
| BTC/USDT | 15m | Bollinger Mean Reversion | fixed_atr | high | -179.913175 |
| BTC/USDT | 15m | Bollinger Mean Reversion | fixed_atr | low | -160.403040 |
| BTC/USDT | 15m | Bollinger Mean Reversion | fixed_atr | realistic | -160.403040 |
| BTC/USDT | 15m | Bollinger Mean Reversion | structure_trailing | high | -150.947350 |
| BTC/USDT | 15m | Bollinger Mean Reversion | structure_trailing | low | -87.362727 |
| BTC/USDT | 15m | Bollinger Mean Reversion | structure_trailing | realistic | -87.362727 |
| BTC/USDT | 15m | Bollinger Mean Reversion | time_based | high | -275.494469 |
| BTC/USDT | 15m | Bollinger Mean Reversion | time_based | low | -222.632101 |
| BTC/USDT | 15m | Bollinger Mean Reversion | time_based | realistic | -222.632101 |
| BTC/USDT | 15m | Bollinger SnapBack | fixed_atr | high | -179.913175 |
| BTC/USDT | 15m | Bollinger SnapBack | fixed_atr | low | -160.403040 |
| BTC/USDT | 15m | Bollinger SnapBack | fixed_atr | realistic | -160.403040 |
| BTC/USDT | 15m | Bollinger SnapBack | structure_trailing | high | -150.947350 |
| BTC/USDT | 15m | Bollinger SnapBack | structure_trailing | low | -87.362727 |
| BTC/USDT | 15m | Bollinger SnapBack | structure_trailing | realistic | -87.362727 |
| BTC/USDT | 15m | Bollinger SnapBack | time_based | high | -275.494469 |
| BTC/USDT | 15m | Bollinger SnapBack | time_based | low | -222.632101 |
| BTC/USDT | 15m | Bollinger SnapBack | time_based | realistic | -222.632101 |
| BTC/USDT | 15m | Breakout Retest | fixed_atr | high | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | fixed_atr | low | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | structure_trailing | high | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | structure_trailing | low | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | time_based | high | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | time_based | low | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | time_based | realistic | 0.000000 |
| BTC/USDT | 15m | Donchian Breakout | fixed_atr | high | -34.525476 |
| BTC/USDT | 15m | Donchian Breakout | fixed_atr | low | -77.080929 |
| BTC/USDT | 15m | Donchian Breakout | fixed_atr | realistic | -77.080929 |
| BTC/USDT | 15m | Donchian Breakout | structure_trailing | high | -29.108385 |
| BTC/USDT | 15m | Donchian Breakout | structure_trailing | low | -71.158005 |
| BTC/USDT | 15m | Donchian Breakout | structure_trailing | realistic | -71.158005 |
| BTC/USDT | 15m | Donchian Breakout | time_based | high | 214.270150 |
| BTC/USDT | 15m | Donchian Breakout | time_based | low | 161.671434 |
| BTC/USDT | 15m | Donchian Breakout | time_based | realistic | 161.671434 |
| BTC/USDT | 15m | MA SlopePullback | fixed_atr | high | -111.238646 |
| BTC/USDT | 15m | MA SlopePullback | fixed_atr | low | -107.860043 |
| BTC/USDT | 15m | MA SlopePullback | fixed_atr | realistic | -107.860043 |
| BTC/USDT | 15m | MA SlopePullback | structure_trailing | high | -120.617925 |
| BTC/USDT | 15m | MA SlopePullback | structure_trailing | low | -131.819405 |
| BTC/USDT | 15m | MA SlopePullback | structure_trailing | realistic | -131.819405 |
| BTC/USDT | 15m | MA SlopePullback | time_based | high | -142.808288 |
| BTC/USDT | 15m | MA SlopePullback | time_based | low | -104.638697 |
| BTC/USDT | 15m | MA SlopePullback | time_based | realistic | -104.638697 |
| BTC/USDT | 15m | RSI Mean Reversion | fixed_atr | high | -179.913175 |
| BTC/USDT | 15m | RSI Mean Reversion | fixed_atr | low | -160.403040 |
| BTC/USDT | 15m | RSI Mean Reversion | fixed_atr | realistic | -160.403040 |
| BTC/USDT | 15m | RSI Mean Reversion | structure_trailing | high | -150.947350 |
| BTC/USDT | 15m | RSI Mean Reversion | structure_trailing | low | -87.362727 |
| BTC/USDT | 15m | RSI Mean Reversion | structure_trailing | realistic | -87.362727 |
| BTC/USDT | 15m | RSI Mean Reversion | time_based | high | -275.494469 |
| BTC/USDT | 15m | RSI Mean Reversion | time_based | low | -222.632101 |
| BTC/USDT | 15m | RSI Mean Reversion | time_based | realistic | -222.632101 |
| BTC/USDT | 15m | Range Breakout w/ EMA Trend Filter | fixed_atr | high | 277.360979 |
| BTC/USDT | 15m | Range Breakout w/ EMA Trend Filter | fixed_atr | low | 239.702756 |
| BTC/USDT | 15m | Range Breakout w/ EMA Trend Filter | fixed_atr | realistic | 239.702756 |
| BTC/USDT | 15m | Range Breakout w/ EMA Trend Filter | structure_trailing | high | -52.959317 |
| BTC/USDT | 15m | Range Breakout w/ EMA Trend Filter | structure_trailing | low | -68.200732 |
| BTC/USDT | 15m | Range Breakout w/ EMA Trend Filter | structure_trailing | realistic | -68.200732 |
| BTC/USDT | 15m | Range Breakout w/ EMA Trend Filter | time_based | high | 277.360979 |
| BTC/USDT | 15m | Range Breakout w/ EMA Trend Filter | time_based | low | 239.702756 |
| BTC/USDT | 15m | Range Breakout w/ EMA Trend Filter | time_based | realistic | 239.702756 |
| BTC/USDT | 15m | Range Fade | fixed_atr | high | 0.000000 |
| BTC/USDT | 15m | Range Fade | fixed_atr | low | 0.000000 |
| BTC/USDT | 15m | Range Fade | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 15m | Range Fade | structure_trailing | high | 0.000000 |
| BTC/USDT | 15m | Range Fade | structure_trailing | low | 0.000000 |
| BTC/USDT | 15m | Range Fade | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 15m | Range Fade | time_based | high | 0.000000 |
| BTC/USDT | 15m | Range Fade | time_based | low | 0.000000 |
| BTC/USDT | 15m | Range Fade | time_based | realistic | 0.000000 |
| BTC/USDT | 15m | Trend Pullback | fixed_atr | high | -104.798396 |
| BTC/USDT | 15m | Trend Pullback | fixed_atr | low | -182.552073 |
| BTC/USDT | 15m | Trend Pullback | fixed_atr | realistic | -182.552073 |
| BTC/USDT | 15m | Trend Pullback | structure_trailing | high | -143.506227 |
| BTC/USDT | 15m | Trend Pullback | structure_trailing | low | -98.106153 |
| BTC/USDT | 15m | Trend Pullback | structure_trailing | realistic | -98.106153 |
| BTC/USDT | 15m | Trend Pullback | time_based | high | -261.130703 |
| BTC/USDT | 15m | Trend Pullback | time_based | low | -272.204134 |
| BTC/USDT | 15m | Trend Pullback | time_based | realistic | -272.204134 |
| BTC/USDT | 15m | Vol Compression Breakout | fixed_atr | high | 0.000000 |
| BTC/USDT | 15m | Vol Compression Breakout | fixed_atr | low | 0.000000 |
| BTC/USDT | 15m | Vol Compression Breakout | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 15m | Vol Compression Breakout | structure_trailing | high | 0.000000 |
| BTC/USDT | 15m | Vol Compression Breakout | structure_trailing | low | 0.000000 |
| BTC/USDT | 15m | Vol Compression Breakout | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 15m | Vol Compression Breakout | time_based | high | 0.000000 |
| BTC/USDT | 15m | Vol Compression Breakout | time_based | low | 0.000000 |
| BTC/USDT | 15m | Vol Compression Breakout | time_based | realistic | 0.000000 |
| BTC/USDT | 1d | ATR Distance Revert | fixed_atr | high | 0.000000 |
| BTC/USDT | 1d | ATR Distance Revert | fixed_atr | low | 0.000000 |
| BTC/USDT | 1d | ATR Distance Revert | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1d | ATR Distance Revert | structure_trailing | high | 0.000000 |
| BTC/USDT | 1d | ATR Distance Revert | structure_trailing | low | 0.000000 |
| BTC/USDT | 1d | ATR Distance Revert | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1d | ATR Distance Revert | time_based | high | 0.000000 |
| BTC/USDT | 1d | ATR Distance Revert | time_based | low | 0.000000 |
| BTC/USDT | 1d | ATR Distance Revert | time_based | realistic | 0.000000 |
| BTC/USDT | 1d | Bollinger Mean Reversion | fixed_atr | high | 0.000000 |
| BTC/USDT | 1d | Bollinger Mean Reversion | fixed_atr | low | 0.000000 |
| BTC/USDT | 1d | Bollinger Mean Reversion | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1d | Bollinger Mean Reversion | structure_trailing | high | 0.000000 |
| BTC/USDT | 1d | Bollinger Mean Reversion | structure_trailing | low | 0.000000 |
| BTC/USDT | 1d | Bollinger Mean Reversion | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1d | Bollinger Mean Reversion | time_based | high | 0.000000 |
| BTC/USDT | 1d | Bollinger Mean Reversion | time_based | low | 0.000000 |
| BTC/USDT | 1d | Bollinger Mean Reversion | time_based | realistic | 0.000000 |
| BTC/USDT | 1d | Bollinger SnapBack | fixed_atr | high | 0.000000 |
| BTC/USDT | 1d | Bollinger SnapBack | fixed_atr | low | 0.000000 |
| BTC/USDT | 1d | Bollinger SnapBack | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1d | Bollinger SnapBack | structure_trailing | high | 0.000000 |
| BTC/USDT | 1d | Bollinger SnapBack | structure_trailing | low | 0.000000 |
| BTC/USDT | 1d | Bollinger SnapBack | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1d | Bollinger SnapBack | time_based | high | 0.000000 |
| BTC/USDT | 1d | Bollinger SnapBack | time_based | low | 0.000000 |
| BTC/USDT | 1d | Bollinger SnapBack | time_based | realistic | 0.000000 |
| BTC/USDT | 1d | Breakout Retest | fixed_atr | high | 0.000000 |
| BTC/USDT | 1d | Breakout Retest | fixed_atr | low | 0.000000 |
| BTC/USDT | 1d | Breakout Retest | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1d | Breakout Retest | structure_trailing | high | 0.000000 |
| BTC/USDT | 1d | Breakout Retest | structure_trailing | low | 0.000000 |
| BTC/USDT | 1d | Breakout Retest | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1d | Breakout Retest | time_based | high | 0.000000 |
| BTC/USDT | 1d | Breakout Retest | time_based | low | 0.000000 |
| BTC/USDT | 1d | Breakout Retest | time_based | realistic | 0.000000 |
| BTC/USDT | 1d | Donchian Breakout | fixed_atr | high | 0.000000 |
| BTC/USDT | 1d | Donchian Breakout | fixed_atr | low | 0.000000 |
| BTC/USDT | 1d | Donchian Breakout | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1d | Donchian Breakout | structure_trailing | high | 0.000000 |
| BTC/USDT | 1d | Donchian Breakout | structure_trailing | low | 0.000000 |
| BTC/USDT | 1d | Donchian Breakout | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1d | Donchian Breakout | time_based | high | 0.000000 |
| BTC/USDT | 1d | Donchian Breakout | time_based | low | 0.000000 |
| BTC/USDT | 1d | Donchian Breakout | time_based | realistic | 0.000000 |
| BTC/USDT | 1d | MA SlopePullback | fixed_atr | high | 0.000000 |
| BTC/USDT | 1d | MA SlopePullback | fixed_atr | low | 0.000000 |
| BTC/USDT | 1d | MA SlopePullback | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1d | MA SlopePullback | structure_trailing | high | 0.000000 |
| BTC/USDT | 1d | MA SlopePullback | structure_trailing | low | 0.000000 |
| BTC/USDT | 1d | MA SlopePullback | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1d | MA SlopePullback | time_based | high | 0.000000 |
| BTC/USDT | 1d | MA SlopePullback | time_based | low | 0.000000 |
| BTC/USDT | 1d | MA SlopePullback | time_based | realistic | 0.000000 |
| BTC/USDT | 1d | RSI Mean Reversion | fixed_atr | high | 0.000000 |
| BTC/USDT | 1d | RSI Mean Reversion | fixed_atr | low | 0.000000 |
| BTC/USDT | 1d | RSI Mean Reversion | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1d | RSI Mean Reversion | structure_trailing | high | 0.000000 |
| BTC/USDT | 1d | RSI Mean Reversion | structure_trailing | low | 0.000000 |
| BTC/USDT | 1d | RSI Mean Reversion | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1d | RSI Mean Reversion | time_based | high | 0.000000 |
| BTC/USDT | 1d | RSI Mean Reversion | time_based | low | 0.000000 |
| BTC/USDT | 1d | RSI Mean Reversion | time_based | realistic | 0.000000 |
| BTC/USDT | 1d | Range Breakout w/ EMA Trend Filter | fixed_atr | high | 0.000000 |
| BTC/USDT | 1d | Range Breakout w/ EMA Trend Filter | fixed_atr | low | 0.000000 |
| BTC/USDT | 1d | Range Breakout w/ EMA Trend Filter | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1d | Range Breakout w/ EMA Trend Filter | structure_trailing | high | 0.000000 |
| BTC/USDT | 1d | Range Breakout w/ EMA Trend Filter | structure_trailing | low | 0.000000 |
| BTC/USDT | 1d | Range Breakout w/ EMA Trend Filter | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1d | Range Breakout w/ EMA Trend Filter | time_based | high | 0.000000 |
| BTC/USDT | 1d | Range Breakout w/ EMA Trend Filter | time_based | low | 0.000000 |
| BTC/USDT | 1d | Range Breakout w/ EMA Trend Filter | time_based | realistic | 0.000000 |
| BTC/USDT | 1d | Range Fade | fixed_atr | high | 0.000000 |
| BTC/USDT | 1d | Range Fade | fixed_atr | low | 0.000000 |
| BTC/USDT | 1d | Range Fade | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1d | Range Fade | structure_trailing | high | 0.000000 |
| BTC/USDT | 1d | Range Fade | structure_trailing | low | 0.000000 |
| BTC/USDT | 1d | Range Fade | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1d | Range Fade | time_based | high | 0.000000 |
| BTC/USDT | 1d | Range Fade | time_based | low | 0.000000 |
| BTC/USDT | 1d | Range Fade | time_based | realistic | 0.000000 |
| BTC/USDT | 1d | Trend Pullback | fixed_atr | high | 0.000000 |
| BTC/USDT | 1d | Trend Pullback | fixed_atr | low | 0.000000 |
| BTC/USDT | 1d | Trend Pullback | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1d | Trend Pullback | structure_trailing | high | 0.000000 |
| BTC/USDT | 1d | Trend Pullback | structure_trailing | low | 0.000000 |
| BTC/USDT | 1d | Trend Pullback | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1d | Trend Pullback | time_based | high | 0.000000 |
| BTC/USDT | 1d | Trend Pullback | time_based | low | 0.000000 |
| BTC/USDT | 1d | Trend Pullback | time_based | realistic | 0.000000 |
| BTC/USDT | 1d | Vol Compression Breakout | fixed_atr | high | 0.000000 |
| BTC/USDT | 1d | Vol Compression Breakout | fixed_atr | low | 0.000000 |
| BTC/USDT | 1d | Vol Compression Breakout | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1d | Vol Compression Breakout | structure_trailing | high | 0.000000 |
| BTC/USDT | 1d | Vol Compression Breakout | structure_trailing | low | 0.000000 |
| BTC/USDT | 1d | Vol Compression Breakout | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1d | Vol Compression Breakout | time_based | high | 0.000000 |
| BTC/USDT | 1d | Vol Compression Breakout | time_based | low | 0.000000 |
| BTC/USDT | 1d | Vol Compression Breakout | time_based | realistic | 0.000000 |
| BTC/USDT | 1h | ATR Distance Revert | fixed_atr | high | 77.349822 |
| BTC/USDT | 1h | ATR Distance Revert | fixed_atr | low | 67.692952 |
| BTC/USDT | 1h | ATR Distance Revert | fixed_atr | realistic | 67.692952 |
| BTC/USDT | 1h | ATR Distance Revert | structure_trailing | high | -12.729487 |
| BTC/USDT | 1h | ATR Distance Revert | structure_trailing | low | -73.315702 |
| BTC/USDT | 1h | ATR Distance Revert | structure_trailing | realistic | -73.315702 |
| BTC/USDT | 1h | ATR Distance Revert | time_based | high | 1045.445099 |
| BTC/USDT | 1h | ATR Distance Revert | time_based | low | 921.700775 |
| BTC/USDT | 1h | ATR Distance Revert | time_based | realistic | 921.700775 |
| BTC/USDT | 1h | Bollinger Mean Reversion | fixed_atr | high | 77.349822 |
| BTC/USDT | 1h | Bollinger Mean Reversion | fixed_atr | low | 169.729342 |
| BTC/USDT | 1h | Bollinger Mean Reversion | fixed_atr | realistic | 169.729342 |
| BTC/USDT | 1h | Bollinger Mean Reversion | structure_trailing | high | -12.729487 |
| BTC/USDT | 1h | Bollinger Mean Reversion | structure_trailing | low | -73.415962 |
| BTC/USDT | 1h | Bollinger Mean Reversion | structure_trailing | realistic | -73.415962 |
| BTC/USDT | 1h | Bollinger Mean Reversion | time_based | high | 1045.445099 |
| BTC/USDT | 1h | Bollinger Mean Reversion | time_based | low | 921.700775 |
| BTC/USDT | 1h | Bollinger Mean Reversion | time_based | realistic | 921.700775 |
| BTC/USDT | 1h | Bollinger SnapBack | fixed_atr | high | 100.417757 |
| BTC/USDT | 1h | Bollinger SnapBack | fixed_atr | low | 80.073434 |
| BTC/USDT | 1h | Bollinger SnapBack | fixed_atr | realistic | 80.073434 |
| BTC/USDT | 1h | Bollinger SnapBack | structure_trailing | high | 110.501671 |
| BTC/USDT | 1h | Bollinger SnapBack | structure_trailing | low | -10.718440 |
| BTC/USDT | 1h | Bollinger SnapBack | structure_trailing | realistic | -10.718440 |
| BTC/USDT | 1h | Bollinger SnapBack | time_based | high | 295.990984 |
| BTC/USDT | 1h | Bollinger SnapBack | time_based | low | 143.406866 |
| BTC/USDT | 1h | Bollinger SnapBack | time_based | realistic | 143.406866 |
| BTC/USDT | 1h | Breakout Retest | fixed_atr | high | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | fixed_atr | low | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | structure_trailing | high | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | structure_trailing | low | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | time_based | high | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | time_based | low | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | time_based | realistic | 0.000000 |
| BTC/USDT | 1h | Donchian Breakout | fixed_atr | high | -275.464121 |
| BTC/USDT | 1h | Donchian Breakout | fixed_atr | low | -295.677068 |
| BTC/USDT | 1h | Donchian Breakout | fixed_atr | realistic | -295.677068 |
| BTC/USDT | 1h | Donchian Breakout | structure_trailing | high | -282.436684 |
| BTC/USDT | 1h | Donchian Breakout | structure_trailing | low | -166.736907 |
| BTC/USDT | 1h | Donchian Breakout | structure_trailing | realistic | -166.736907 |
| BTC/USDT | 1h | Donchian Breakout | time_based | high | -778.023150 |
| BTC/USDT | 1h | Donchian Breakout | time_based | low | -663.853865 |
| BTC/USDT | 1h | Donchian Breakout | time_based | realistic | -663.853865 |
| BTC/USDT | 1h | MA SlopePullback | fixed_atr | high | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | fixed_atr | low | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | structure_trailing | high | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | structure_trailing | low | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | time_based | high | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | time_based | low | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | time_based | realistic | 0.000000 |
| BTC/USDT | 1h | RSI Mean Reversion | fixed_atr | high | 80.036681 |
| BTC/USDT | 1h | RSI Mean Reversion | fixed_atr | low | 162.144352 |
| BTC/USDT | 1h | RSI Mean Reversion | fixed_atr | realistic | 162.144352 |
| BTC/USDT | 1h | RSI Mean Reversion | structure_trailing | high | 20.055435 |
| BTC/USDT | 1h | RSI Mean Reversion | structure_trailing | low | -25.237179 |
| BTC/USDT | 1h | RSI Mean Reversion | structure_trailing | realistic | -25.237179 |
| BTC/USDT | 1h | RSI Mean Reversion | time_based | high | 1064.693271 |
| BTC/USDT | 1h | RSI Mean Reversion | time_based | low | 940.733404 |
| BTC/USDT | 1h | RSI Mean Reversion | time_based | realistic | 940.733404 |
| BTC/USDT | 1h | Range Breakout w/ EMA Trend Filter | fixed_atr | high | 0.000000 |
| BTC/USDT | 1h | Range Breakout w/ EMA Trend Filter | fixed_atr | low | 0.000000 |
| BTC/USDT | 1h | Range Breakout w/ EMA Trend Filter | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1h | Range Breakout w/ EMA Trend Filter | structure_trailing | high | 0.000000 |
| BTC/USDT | 1h | Range Breakout w/ EMA Trend Filter | structure_trailing | low | 0.000000 |
| BTC/USDT | 1h | Range Breakout w/ EMA Trend Filter | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1h | Range Breakout w/ EMA Trend Filter | time_based | high | 0.000000 |
| BTC/USDT | 1h | Range Breakout w/ EMA Trend Filter | time_based | low | 0.000000 |
| BTC/USDT | 1h | Range Breakout w/ EMA Trend Filter | time_based | realistic | 0.000000 |
| BTC/USDT | 1h | Range Fade | fixed_atr | high | 0.000000 |
| BTC/USDT | 1h | Range Fade | fixed_atr | low | 0.000000 |
| BTC/USDT | 1h | Range Fade | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1h | Range Fade | structure_trailing | high | 0.000000 |
| BTC/USDT | 1h | Range Fade | structure_trailing | low | 0.000000 |
| BTC/USDT | 1h | Range Fade | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1h | Range Fade | time_based | high | 0.000000 |
| BTC/USDT | 1h | Range Fade | time_based | low | 0.000000 |
| BTC/USDT | 1h | Range Fade | time_based | realistic | 0.000000 |
| BTC/USDT | 1h | Trend Pullback | fixed_atr | high | 0.000000 |
| BTC/USDT | 1h | Trend Pullback | fixed_atr | low | 0.000000 |
| BTC/USDT | 1h | Trend Pullback | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1h | Trend Pullback | structure_trailing | high | 0.000000 |
| BTC/USDT | 1h | Trend Pullback | structure_trailing | low | 0.000000 |
| BTC/USDT | 1h | Trend Pullback | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1h | Trend Pullback | time_based | high | 0.000000 |
| BTC/USDT | 1h | Trend Pullback | time_based | low | 0.000000 |
| BTC/USDT | 1h | Trend Pullback | time_based | realistic | 0.000000 |
| BTC/USDT | 1h | Vol Compression Breakout | fixed_atr | high | 0.000000 |
| BTC/USDT | 1h | Vol Compression Breakout | fixed_atr | low | 0.000000 |
| BTC/USDT | 1h | Vol Compression Breakout | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1h | Vol Compression Breakout | structure_trailing | high | 0.000000 |
| BTC/USDT | 1h | Vol Compression Breakout | structure_trailing | low | 0.000000 |
| BTC/USDT | 1h | Vol Compression Breakout | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1h | Vol Compression Breakout | time_based | high | 0.000000 |
| BTC/USDT | 1h | Vol Compression Breakout | time_based | low | 0.000000 |
| BTC/USDT | 1h | Vol Compression Breakout | time_based | realistic | 0.000000 |
| BTC/USDT | 2h | ATR Distance Revert | fixed_atr | high | 0.000000 |
| BTC/USDT | 2h | ATR Distance Revert | fixed_atr | low | 0.000000 |
| BTC/USDT | 2h | ATR Distance Revert | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 2h | ATR Distance Revert | structure_trailing | high | 0.000000 |
| BTC/USDT | 2h | ATR Distance Revert | structure_trailing | low | 0.000000 |
| BTC/USDT | 2h | ATR Distance Revert | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 2h | ATR Distance Revert | time_based | high | 0.000000 |
| BTC/USDT | 2h | ATR Distance Revert | time_based | low | 0.000000 |
| BTC/USDT | 2h | ATR Distance Revert | time_based | realistic | 0.000000 |
| BTC/USDT | 2h | Bollinger Mean Reversion | fixed_atr | high | 0.000000 |
| BTC/USDT | 2h | Bollinger Mean Reversion | fixed_atr | low | 0.000000 |
| BTC/USDT | 2h | Bollinger Mean Reversion | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 2h | Bollinger Mean Reversion | structure_trailing | high | 0.000000 |
| BTC/USDT | 2h | Bollinger Mean Reversion | structure_trailing | low | 0.000000 |
| BTC/USDT | 2h | Bollinger Mean Reversion | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 2h | Bollinger Mean Reversion | time_based | high | 0.000000 |
| BTC/USDT | 2h | Bollinger Mean Reversion | time_based | low | 0.000000 |
| BTC/USDT | 2h | Bollinger Mean Reversion | time_based | realistic | 0.000000 |
| BTC/USDT | 2h | Bollinger SnapBack | fixed_atr | high | 0.000000 |
| BTC/USDT | 2h | Bollinger SnapBack | fixed_atr | low | 0.000000 |
| BTC/USDT | 2h | Bollinger SnapBack | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 2h | Bollinger SnapBack | structure_trailing | high | 0.000000 |
| BTC/USDT | 2h | Bollinger SnapBack | structure_trailing | low | 0.000000 |
| BTC/USDT | 2h | Bollinger SnapBack | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 2h | Bollinger SnapBack | time_based | high | 0.000000 |
| BTC/USDT | 2h | Bollinger SnapBack | time_based | low | 0.000000 |
| BTC/USDT | 2h | Bollinger SnapBack | time_based | realistic | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | fixed_atr | high | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | fixed_atr | low | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | structure_trailing | high | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | structure_trailing | low | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | time_based | high | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | time_based | low | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | time_based | realistic | 0.000000 |
| BTC/USDT | 2h | Donchian Breakout | fixed_atr | high | 0.000000 |
| BTC/USDT | 2h | Donchian Breakout | fixed_atr | low | 0.000000 |
| BTC/USDT | 2h | Donchian Breakout | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 2h | Donchian Breakout | structure_trailing | high | 0.000000 |
| BTC/USDT | 2h | Donchian Breakout | structure_trailing | low | 0.000000 |
| BTC/USDT | 2h | Donchian Breakout | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 2h | Donchian Breakout | time_based | high | 0.000000 |
| BTC/USDT | 2h | Donchian Breakout | time_based | low | 0.000000 |
| BTC/USDT | 2h | Donchian Breakout | time_based | realistic | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | fixed_atr | high | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | fixed_atr | low | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | structure_trailing | high | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | structure_trailing | low | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | time_based | high | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | time_based | low | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | time_based | realistic | 0.000000 |
| BTC/USDT | 2h | RSI Mean Reversion | fixed_atr | high | 0.000000 |
| BTC/USDT | 2h | RSI Mean Reversion | fixed_atr | low | 0.000000 |
| BTC/USDT | 2h | RSI Mean Reversion | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 2h | RSI Mean Reversion | structure_trailing | high | 0.000000 |
| BTC/USDT | 2h | RSI Mean Reversion | structure_trailing | low | 0.000000 |
| BTC/USDT | 2h | RSI Mean Reversion | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 2h | RSI Mean Reversion | time_based | high | 0.000000 |
| BTC/USDT | 2h | RSI Mean Reversion | time_based | low | 0.000000 |
| BTC/USDT | 2h | RSI Mean Reversion | time_based | realistic | 0.000000 |
| BTC/USDT | 2h | Range Breakout w/ EMA Trend Filter | fixed_atr | high | 0.000000 |
| BTC/USDT | 2h | Range Breakout w/ EMA Trend Filter | fixed_atr | low | 0.000000 |
| BTC/USDT | 2h | Range Breakout w/ EMA Trend Filter | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 2h | Range Breakout w/ EMA Trend Filter | structure_trailing | high | 0.000000 |
| BTC/USDT | 2h | Range Breakout w/ EMA Trend Filter | structure_trailing | low | 0.000000 |
| BTC/USDT | 2h | Range Breakout w/ EMA Trend Filter | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 2h | Range Breakout w/ EMA Trend Filter | time_based | high | 0.000000 |
| BTC/USDT | 2h | Range Breakout w/ EMA Trend Filter | time_based | low | 0.000000 |
| BTC/USDT | 2h | Range Breakout w/ EMA Trend Filter | time_based | realistic | 0.000000 |
| BTC/USDT | 2h | Range Fade | fixed_atr | high | 0.000000 |
| BTC/USDT | 2h | Range Fade | fixed_atr | low | 0.000000 |
| BTC/USDT | 2h | Range Fade | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 2h | Range Fade | structure_trailing | high | 0.000000 |
| BTC/USDT | 2h | Range Fade | structure_trailing | low | 0.000000 |
| BTC/USDT | 2h | Range Fade | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 2h | Range Fade | time_based | high | 0.000000 |
| BTC/USDT | 2h | Range Fade | time_based | low | 0.000000 |
| BTC/USDT | 2h | Range Fade | time_based | realistic | 0.000000 |
| BTC/USDT | 2h | Trend Pullback | fixed_atr | high | 0.000000 |
| BTC/USDT | 2h | Trend Pullback | fixed_atr | low | 0.000000 |
| BTC/USDT | 2h | Trend Pullback | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 2h | Trend Pullback | structure_trailing | high | 0.000000 |
| BTC/USDT | 2h | Trend Pullback | structure_trailing | low | 0.000000 |
| BTC/USDT | 2h | Trend Pullback | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 2h | Trend Pullback | time_based | high | 0.000000 |
| BTC/USDT | 2h | Trend Pullback | time_based | low | 0.000000 |
| BTC/USDT | 2h | Trend Pullback | time_based | realistic | 0.000000 |
| BTC/USDT | 2h | Vol Compression Breakout | fixed_atr | high | 0.000000 |
| BTC/USDT | 2h | Vol Compression Breakout | fixed_atr | low | 0.000000 |
| BTC/USDT | 2h | Vol Compression Breakout | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 2h | Vol Compression Breakout | structure_trailing | high | 0.000000 |
| BTC/USDT | 2h | Vol Compression Breakout | structure_trailing | low | 0.000000 |
| BTC/USDT | 2h | Vol Compression Breakout | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 2h | Vol Compression Breakout | time_based | high | 0.000000 |
| BTC/USDT | 2h | Vol Compression Breakout | time_based | low | 0.000000 |
| BTC/USDT | 2h | Vol Compression Breakout | time_based | realistic | 0.000000 |
| BTC/USDT | 30m | ATR Distance Revert | fixed_atr | high | 279.899708 |
| BTC/USDT | 30m | ATR Distance Revert | fixed_atr | low | 436.484568 |
| BTC/USDT | 30m | ATR Distance Revert | fixed_atr | realistic | 436.484568 |
| BTC/USDT | 30m | ATR Distance Revert | structure_trailing | high | -197.965630 |
| BTC/USDT | 30m | ATR Distance Revert | structure_trailing | low | -174.203939 |
| BTC/USDT | 30m | ATR Distance Revert | structure_trailing | realistic | -174.203939 |
| BTC/USDT | 30m | ATR Distance Revert | time_based | high | 312.206090 |
| BTC/USDT | 30m | ATR Distance Revert | time_based | low | 254.720682 |
| BTC/USDT | 30m | ATR Distance Revert | time_based | realistic | 254.720682 |
| BTC/USDT | 30m | Bollinger Mean Reversion | fixed_atr | high | -106.706400 |
| BTC/USDT | 30m | Bollinger Mean Reversion | fixed_atr | low | -74.947729 |
| BTC/USDT | 30m | Bollinger Mean Reversion | fixed_atr | realistic | -74.947729 |
| BTC/USDT | 30m | Bollinger Mean Reversion | structure_trailing | high | -16.801737 |
| BTC/USDT | 30m | Bollinger Mean Reversion | structure_trailing | low | 24.999176 |
| BTC/USDT | 30m | Bollinger Mean Reversion | structure_trailing | realistic | 24.999176 |
| BTC/USDT | 30m | Bollinger Mean Reversion | time_based | high | 60.558581 |
| BTC/USDT | 30m | Bollinger Mean Reversion | time_based | low | 5.166854 |
| BTC/USDT | 30m | Bollinger Mean Reversion | time_based | realistic | 5.166854 |
| BTC/USDT | 30m | Bollinger SnapBack | fixed_atr | high | -106.706400 |
| BTC/USDT | 30m | Bollinger SnapBack | fixed_atr | low | -74.947729 |
| BTC/USDT | 30m | Bollinger SnapBack | fixed_atr | realistic | -74.947729 |
| BTC/USDT | 30m | Bollinger SnapBack | structure_trailing | high | -16.801737 |
| BTC/USDT | 30m | Bollinger SnapBack | structure_trailing | low | 24.999176 |
| BTC/USDT | 30m | Bollinger SnapBack | structure_trailing | realistic | 24.999176 |
| BTC/USDT | 30m | Bollinger SnapBack | time_based | high | 60.558581 |
| BTC/USDT | 30m | Bollinger SnapBack | time_based | low | 5.166854 |
| BTC/USDT | 30m | Bollinger SnapBack | time_based | realistic | 5.166854 |
| BTC/USDT | 30m | Breakout Retest | fixed_atr | high | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | fixed_atr | low | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | structure_trailing | high | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | structure_trailing | low | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | time_based | high | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | time_based | low | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | time_based | realistic | 0.000000 |
| BTC/USDT | 30m | Donchian Breakout | fixed_atr | high | -248.796205 |
| BTC/USDT | 30m | Donchian Breakout | fixed_atr | low | -248.044574 |
| BTC/USDT | 30m | Donchian Breakout | fixed_atr | realistic | -248.044574 |
| BTC/USDT | 30m | Donchian Breakout | structure_trailing | high | -133.280907 |
| BTC/USDT | 30m | Donchian Breakout | structure_trailing | low | -125.222980 |
| BTC/USDT | 30m | Donchian Breakout | structure_trailing | realistic | -125.222980 |
| BTC/USDT | 30m | Donchian Breakout | time_based | high | -491.653117 |
| BTC/USDT | 30m | Donchian Breakout | time_based | low | -441.150231 |
| BTC/USDT | 30m | Donchian Breakout | time_based | realistic | -441.150231 |
| BTC/USDT | 30m | MA SlopePullback | fixed_atr | high | -272.989820 |
| BTC/USDT | 30m | MA SlopePullback | fixed_atr | low | -218.348127 |
| BTC/USDT | 30m | MA SlopePullback | fixed_atr | realistic | -218.348127 |
| BTC/USDT | 30m | MA SlopePullback | structure_trailing | high | -138.884222 |
| BTC/USDT | 30m | MA SlopePullback | structure_trailing | low | -130.520520 |
| BTC/USDT | 30m | MA SlopePullback | structure_trailing | realistic | -130.520520 |
| BTC/USDT | 30m | MA SlopePullback | time_based | high | -470.746298 |
| BTC/USDT | 30m | MA SlopePullback | time_based | low | -396.672847 |
| BTC/USDT | 30m | MA SlopePullback | time_based | realistic | -396.672847 |
| BTC/USDT | 30m | RSI Mean Reversion | fixed_atr | high | 279.899708 |
| BTC/USDT | 30m | RSI Mean Reversion | fixed_atr | low | 436.484568 |
| BTC/USDT | 30m | RSI Mean Reversion | fixed_atr | realistic | 436.484568 |
| BTC/USDT | 30m | RSI Mean Reversion | structure_trailing | high | -197.965630 |
| BTC/USDT | 30m | RSI Mean Reversion | structure_trailing | low | -174.203939 |
| BTC/USDT | 30m | RSI Mean Reversion | structure_trailing | realistic | -174.203939 |
| BTC/USDT | 30m | RSI Mean Reversion | time_based | high | 312.206090 |
| BTC/USDT | 30m | RSI Mean Reversion | time_based | low | 254.720682 |
| BTC/USDT | 30m | RSI Mean Reversion | time_based | realistic | 254.720682 |
| BTC/USDT | 30m | Range Breakout w/ EMA Trend Filter | fixed_atr | high | 0.000000 |
| BTC/USDT | 30m | Range Breakout w/ EMA Trend Filter | fixed_atr | low | 0.000000 |
| BTC/USDT | 30m | Range Breakout w/ EMA Trend Filter | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 30m | Range Breakout w/ EMA Trend Filter | structure_trailing | high | 0.000000 |
| BTC/USDT | 30m | Range Breakout w/ EMA Trend Filter | structure_trailing | low | 0.000000 |
| BTC/USDT | 30m | Range Breakout w/ EMA Trend Filter | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 30m | Range Breakout w/ EMA Trend Filter | time_based | high | 0.000000 |
| BTC/USDT | 30m | Range Breakout w/ EMA Trend Filter | time_based | low | 0.000000 |
| BTC/USDT | 30m | Range Breakout w/ EMA Trend Filter | time_based | realistic | 0.000000 |
| BTC/USDT | 30m | Range Fade | fixed_atr | high | 0.000000 |
| BTC/USDT | 30m | Range Fade | fixed_atr | low | 0.000000 |
| BTC/USDT | 30m | Range Fade | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 30m | Range Fade | structure_trailing | high | 0.000000 |
| BTC/USDT | 30m | Range Fade | structure_trailing | low | 0.000000 |
| BTC/USDT | 30m | Range Fade | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 30m | Range Fade | time_based | high | 0.000000 |
| BTC/USDT | 30m | Range Fade | time_based | low | 0.000000 |
| BTC/USDT | 30m | Range Fade | time_based | realistic | 0.000000 |
| BTC/USDT | 30m | Trend Pullback | fixed_atr | high | 0.000000 |
| BTC/USDT | 30m | Trend Pullback | fixed_atr | low | 0.000000 |
| BTC/USDT | 30m | Trend Pullback | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 30m | Trend Pullback | structure_trailing | high | 0.000000 |
| BTC/USDT | 30m | Trend Pullback | structure_trailing | low | 0.000000 |
| BTC/USDT | 30m | Trend Pullback | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 30m | Trend Pullback | time_based | high | 0.000000 |
| BTC/USDT | 30m | Trend Pullback | time_based | low | 0.000000 |
| BTC/USDT | 30m | Trend Pullback | time_based | realistic | 0.000000 |
| BTC/USDT | 30m | Vol Compression Breakout | fixed_atr | high | 0.000000 |
| BTC/USDT | 30m | Vol Compression Breakout | fixed_atr | low | 0.000000 |
| BTC/USDT | 30m | Vol Compression Breakout | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 30m | Vol Compression Breakout | structure_trailing | high | 0.000000 |
| BTC/USDT | 30m | Vol Compression Breakout | structure_trailing | low | 0.000000 |
| BTC/USDT | 30m | Vol Compression Breakout | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 30m | Vol Compression Breakout | time_based | high | 0.000000 |
| BTC/USDT | 30m | Vol Compression Breakout | time_based | low | 0.000000 |
| BTC/USDT | 30m | Vol Compression Breakout | time_based | realistic | 0.000000 |
| BTC/USDT | 4h | ATR Distance Revert | fixed_atr | high | 0.000000 |
| BTC/USDT | 4h | ATR Distance Revert | fixed_atr | low | 0.000000 |
| BTC/USDT | 4h | ATR Distance Revert | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 4h | ATR Distance Revert | structure_trailing | high | 0.000000 |
| BTC/USDT | 4h | ATR Distance Revert | structure_trailing | low | 0.000000 |
| BTC/USDT | 4h | ATR Distance Revert | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 4h | ATR Distance Revert | time_based | high | 0.000000 |
| BTC/USDT | 4h | ATR Distance Revert | time_based | low | 0.000000 |
| BTC/USDT | 4h | ATR Distance Revert | time_based | realistic | 0.000000 |
| BTC/USDT | 4h | Bollinger Mean Reversion | fixed_atr | high | 0.000000 |
| BTC/USDT | 4h | Bollinger Mean Reversion | fixed_atr | low | 0.000000 |
| BTC/USDT | 4h | Bollinger Mean Reversion | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 4h | Bollinger Mean Reversion | structure_trailing | high | 0.000000 |
| BTC/USDT | 4h | Bollinger Mean Reversion | structure_trailing | low | 0.000000 |
| BTC/USDT | 4h | Bollinger Mean Reversion | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 4h | Bollinger Mean Reversion | time_based | high | 0.000000 |
| BTC/USDT | 4h | Bollinger Mean Reversion | time_based | low | 0.000000 |
| BTC/USDT | 4h | Bollinger Mean Reversion | time_based | realistic | 0.000000 |
| BTC/USDT | 4h | Bollinger SnapBack | fixed_atr | high | 0.000000 |
| BTC/USDT | 4h | Bollinger SnapBack | fixed_atr | low | 0.000000 |
| BTC/USDT | 4h | Bollinger SnapBack | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 4h | Bollinger SnapBack | structure_trailing | high | 0.000000 |
| BTC/USDT | 4h | Bollinger SnapBack | structure_trailing | low | 0.000000 |
| BTC/USDT | 4h | Bollinger SnapBack | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 4h | Bollinger SnapBack | time_based | high | 0.000000 |
| BTC/USDT | 4h | Bollinger SnapBack | time_based | low | 0.000000 |
| BTC/USDT | 4h | Bollinger SnapBack | time_based | realistic | 0.000000 |
| BTC/USDT | 4h | Breakout Retest | fixed_atr | high | 0.000000 |
| BTC/USDT | 4h | Breakout Retest | fixed_atr | low | 0.000000 |
| BTC/USDT | 4h | Breakout Retest | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 4h | Breakout Retest | structure_trailing | high | 0.000000 |
| BTC/USDT | 4h | Breakout Retest | structure_trailing | low | 0.000000 |
| BTC/USDT | 4h | Breakout Retest | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 4h | Breakout Retest | time_based | high | 0.000000 |
| BTC/USDT | 4h | Breakout Retest | time_based | low | 0.000000 |
| BTC/USDT | 4h | Breakout Retest | time_based | realistic | 0.000000 |
| BTC/USDT | 4h | Donchian Breakout | fixed_atr | high | 0.000000 |
| BTC/USDT | 4h | Donchian Breakout | fixed_atr | low | 0.000000 |
| BTC/USDT | 4h | Donchian Breakout | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 4h | Donchian Breakout | structure_trailing | high | 0.000000 |
| BTC/USDT | 4h | Donchian Breakout | structure_trailing | low | 0.000000 |
| BTC/USDT | 4h | Donchian Breakout | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 4h | Donchian Breakout | time_based | high | 0.000000 |
| BTC/USDT | 4h | Donchian Breakout | time_based | low | 0.000000 |
| BTC/USDT | 4h | Donchian Breakout | time_based | realistic | 0.000000 |
| BTC/USDT | 4h | MA SlopePullback | fixed_atr | high | 0.000000 |
| BTC/USDT | 4h | MA SlopePullback | fixed_atr | low | 0.000000 |
| BTC/USDT | 4h | MA SlopePullback | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 4h | MA SlopePullback | structure_trailing | high | 0.000000 |
| BTC/USDT | 4h | MA SlopePullback | structure_trailing | low | 0.000000 |
| BTC/USDT | 4h | MA SlopePullback | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 4h | MA SlopePullback | time_based | high | 0.000000 |
| BTC/USDT | 4h | MA SlopePullback | time_based | low | 0.000000 |
| BTC/USDT | 4h | MA SlopePullback | time_based | realistic | 0.000000 |
| BTC/USDT | 4h | RSI Mean Reversion | fixed_atr | high | 0.000000 |
| BTC/USDT | 4h | RSI Mean Reversion | fixed_atr | low | 0.000000 |
| BTC/USDT | 4h | RSI Mean Reversion | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 4h | RSI Mean Reversion | structure_trailing | high | 0.000000 |
| BTC/USDT | 4h | RSI Mean Reversion | structure_trailing | low | 0.000000 |
| BTC/USDT | 4h | RSI Mean Reversion | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 4h | RSI Mean Reversion | time_based | high | 0.000000 |
| BTC/USDT | 4h | RSI Mean Reversion | time_based | low | 0.000000 |
| BTC/USDT | 4h | RSI Mean Reversion | time_based | realistic | 0.000000 |
| BTC/USDT | 4h | Range Breakout w/ EMA Trend Filter | fixed_atr | high | 0.000000 |
| BTC/USDT | 4h | Range Breakout w/ EMA Trend Filter | fixed_atr | low | 0.000000 |
| BTC/USDT | 4h | Range Breakout w/ EMA Trend Filter | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 4h | Range Breakout w/ EMA Trend Filter | structure_trailing | high | 0.000000 |
| BTC/USDT | 4h | Range Breakout w/ EMA Trend Filter | structure_trailing | low | 0.000000 |
| BTC/USDT | 4h | Range Breakout w/ EMA Trend Filter | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 4h | Range Breakout w/ EMA Trend Filter | time_based | high | 0.000000 |
| BTC/USDT | 4h | Range Breakout w/ EMA Trend Filter | time_based | low | 0.000000 |
| BTC/USDT | 4h | Range Breakout w/ EMA Trend Filter | time_based | realistic | 0.000000 |
| BTC/USDT | 4h | Range Fade | fixed_atr | high | 0.000000 |
| BTC/USDT | 4h | Range Fade | fixed_atr | low | 0.000000 |
| BTC/USDT | 4h | Range Fade | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 4h | Range Fade | structure_trailing | high | 0.000000 |
| BTC/USDT | 4h | Range Fade | structure_trailing | low | 0.000000 |
| BTC/USDT | 4h | Range Fade | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 4h | Range Fade | time_based | high | 0.000000 |
| BTC/USDT | 4h | Range Fade | time_based | low | 0.000000 |
| BTC/USDT | 4h | Range Fade | time_based | realistic | 0.000000 |
| BTC/USDT | 4h | Trend Pullback | fixed_atr | high | 0.000000 |
| BTC/USDT | 4h | Trend Pullback | fixed_atr | low | 0.000000 |
| BTC/USDT | 4h | Trend Pullback | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 4h | Trend Pullback | structure_trailing | high | 0.000000 |
| BTC/USDT | 4h | Trend Pullback | structure_trailing | low | 0.000000 |
| BTC/USDT | 4h | Trend Pullback | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 4h | Trend Pullback | time_based | high | 0.000000 |
| BTC/USDT | 4h | Trend Pullback | time_based | low | 0.000000 |
| BTC/USDT | 4h | Trend Pullback | time_based | realistic | 0.000000 |
| BTC/USDT | 4h | Vol Compression Breakout | fixed_atr | high | 0.000000 |
| BTC/USDT | 4h | Vol Compression Breakout | fixed_atr | low | 0.000000 |
| BTC/USDT | 4h | Vol Compression Breakout | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 4h | Vol Compression Breakout | structure_trailing | high | 0.000000 |
| BTC/USDT | 4h | Vol Compression Breakout | structure_trailing | low | 0.000000 |
| BTC/USDT | 4h | Vol Compression Breakout | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 4h | Vol Compression Breakout | time_based | high | 0.000000 |
| BTC/USDT | 4h | Vol Compression Breakout | time_based | low | 0.000000 |
| BTC/USDT | 4h | Vol Compression Breakout | time_based | realistic | 0.000000 |
| ETH/USDT | 15m | ATR Distance Revert | fixed_atr | high | -87.733808 |
| ETH/USDT | 15m | ATR Distance Revert | fixed_atr | low | -99.940172 |
| ETH/USDT | 15m | ATR Distance Revert | fixed_atr | realistic | -99.940172 |
| ETH/USDT | 15m | ATR Distance Revert | structure_trailing | high | -95.008387 |
| ETH/USDT | 15m | ATR Distance Revert | structure_trailing | low | -122.627287 |
| ETH/USDT | 15m | ATR Distance Revert | structure_trailing | realistic | -122.627287 |
| ETH/USDT | 15m | ATR Distance Revert | time_based | high | 275.296791 |
| ETH/USDT | 15m | ATR Distance Revert | time_based | low | 397.769555 |
| ETH/USDT | 15m | ATR Distance Revert | time_based | realistic | 397.769555 |
| ETH/USDT | 15m | Bollinger Mean Reversion | fixed_atr | high | -106.092675 |
| ETH/USDT | 15m | Bollinger Mean Reversion | fixed_atr | low | -118.713327 |
| ETH/USDT | 15m | Bollinger Mean Reversion | fixed_atr | realistic | -118.713327 |
| ETH/USDT | 15m | Bollinger Mean Reversion | structure_trailing | high | -76.276655 |
| ETH/USDT | 15m | Bollinger Mean Reversion | structure_trailing | low | -92.713573 |
| ETH/USDT | 15m | Bollinger Mean Reversion | structure_trailing | realistic | -92.713573 |
| ETH/USDT | 15m | Bollinger Mean Reversion | time_based | high | -479.046484 |
| ETH/USDT | 15m | Bollinger Mean Reversion | time_based | low | -446.520124 |
| ETH/USDT | 15m | Bollinger Mean Reversion | time_based | realistic | -446.520124 |
| ETH/USDT | 15m | Bollinger SnapBack | fixed_atr | high | -121.647849 |
| ETH/USDT | 15m | Bollinger SnapBack | fixed_atr | low | -116.655023 |
| ETH/USDT | 15m | Bollinger SnapBack | fixed_atr | realistic | -116.655023 |
| ETH/USDT | 15m | Bollinger SnapBack | structure_trailing | high | -42.248011 |
| ETH/USDT | 15m | Bollinger SnapBack | structure_trailing | low | -64.568671 |
| ETH/USDT | 15m | Bollinger SnapBack | structure_trailing | realistic | -64.568671 |
| ETH/USDT | 15m | Bollinger SnapBack | time_based | high | -306.822448 |
| ETH/USDT | 15m | Bollinger SnapBack | time_based | low | -315.401636 |
| ETH/USDT | 15m | Bollinger SnapBack | time_based | realistic | -315.401636 |
| ETH/USDT | 15m | Breakout Retest | fixed_atr | high | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | fixed_atr | low | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | structure_trailing | high | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | structure_trailing | low | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | time_based | high | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | time_based | low | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | time_based | realistic | 0.000000 |
| ETH/USDT | 15m | Donchian Breakout | fixed_atr | high | -113.825394 |
| ETH/USDT | 15m | Donchian Breakout | fixed_atr | low | -161.383526 |
| ETH/USDT | 15m | Donchian Breakout | fixed_atr | realistic | -161.383526 |
| ETH/USDT | 15m | Donchian Breakout | structure_trailing | high | -54.501953 |
| ETH/USDT | 15m | Donchian Breakout | structure_trailing | low | -35.099358 |
| ETH/USDT | 15m | Donchian Breakout | structure_trailing | realistic | -35.099358 |
| ETH/USDT | 15m | Donchian Breakout | time_based | high | -343.767987 |
| ETH/USDT | 15m | Donchian Breakout | time_based | low | -398.726511 |
| ETH/USDT | 15m | Donchian Breakout | time_based | realistic | -398.726511 |
| ETH/USDT | 15m | MA SlopePullback | fixed_atr | high | -182.671850 |
| ETH/USDT | 15m | MA SlopePullback | fixed_atr | low | -164.453925 |
| ETH/USDT | 15m | MA SlopePullback | fixed_atr | realistic | -164.453925 |
| ETH/USDT | 15m | MA SlopePullback | structure_trailing | high | -69.381715 |
| ETH/USDT | 15m | MA SlopePullback | structure_trailing | low | -84.920509 |
| ETH/USDT | 15m | MA SlopePullback | structure_trailing | realistic | -84.920509 |
| ETH/USDT | 15m | MA SlopePullback | time_based | high | -262.179706 |
| ETH/USDT | 15m | MA SlopePullback | time_based | low | -316.595095 |
| ETH/USDT | 15m | MA SlopePullback | time_based | realistic | -316.595095 |
| ETH/USDT | 15m | RSI Mean Reversion | fixed_atr | high | -87.186185 |
| ETH/USDT | 15m | RSI Mean Reversion | fixed_atr | low | -83.748090 |
| ETH/USDT | 15m | RSI Mean Reversion | fixed_atr | realistic | -83.748090 |
| ETH/USDT | 15m | RSI Mean Reversion | structure_trailing | high | -77.508596 |
| ETH/USDT | 15m | RSI Mean Reversion | structure_trailing | low | -102.947514 |
| ETH/USDT | 15m | RSI Mean Reversion | structure_trailing | realistic | -102.947514 |
| ETH/USDT | 15m | RSI Mean Reversion | time_based | high | 275.296791 |
| ETH/USDT | 15m | RSI Mean Reversion | time_based | low | 397.769555 |
| ETH/USDT | 15m | RSI Mean Reversion | time_based | realistic | 397.769555 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | fixed_atr | high | -83.821722 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | fixed_atr | low | -121.945991 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | fixed_atr | realistic | -121.945991 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | structure_trailing | high | -50.568163 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | structure_trailing | low | -84.710956 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | structure_trailing | realistic | -84.710956 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | time_based | high | 221.960003 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | time_based | low | 272.326584 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | time_based | realistic | 272.326584 |
| ETH/USDT | 15m | Range Fade | fixed_atr | high | 336.459670 |
| ETH/USDT | 15m | Range Fade | fixed_atr | low | 327.076099 |
| ETH/USDT | 15m | Range Fade | fixed_atr | realistic | 327.076099 |
| ETH/USDT | 15m | Range Fade | structure_trailing | high | 461.040112 |
| ETH/USDT | 15m | Range Fade | structure_trailing | low | 330.411757 |
| ETH/USDT | 15m | Range Fade | structure_trailing | realistic | 330.411757 |
| ETH/USDT | 15m | Range Fade | time_based | high | 664.495842 |
| ETH/USDT | 15m | Range Fade | time_based | low | 614.966697 |
| ETH/USDT | 15m | Range Fade | time_based | realistic | 614.966697 |
| ETH/USDT | 15m | Trend Pullback | fixed_atr | high | -173.923816 |
| ETH/USDT | 15m | Trend Pullback | fixed_atr | low | -184.759520 |
| ETH/USDT | 15m | Trend Pullback | fixed_atr | realistic | -184.759520 |
| ETH/USDT | 15m | Trend Pullback | structure_trailing | high | -173.923816 |
| ETH/USDT | 15m | Trend Pullback | structure_trailing | low | -150.766233 |
| ETH/USDT | 15m | Trend Pullback | structure_trailing | realistic | -150.766233 |
| ETH/USDT | 15m | Trend Pullback | time_based | high | 387.014496 |
| ETH/USDT | 15m | Trend Pullback | time_based | low | 378.641365 |
| ETH/USDT | 15m | Trend Pullback | time_based | realistic | 378.641365 |
| ETH/USDT | 15m | Vol Compression Breakout | fixed_atr | high | 0.000000 |
| ETH/USDT | 15m | Vol Compression Breakout | fixed_atr | low | 0.000000 |
| ETH/USDT | 15m | Vol Compression Breakout | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 15m | Vol Compression Breakout | structure_trailing | high | 0.000000 |
| ETH/USDT | 15m | Vol Compression Breakout | structure_trailing | low | 0.000000 |
| ETH/USDT | 15m | Vol Compression Breakout | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 15m | Vol Compression Breakout | time_based | high | 0.000000 |
| ETH/USDT | 15m | Vol Compression Breakout | time_based | low | 0.000000 |
| ETH/USDT | 15m | Vol Compression Breakout | time_based | realistic | 0.000000 |
| ETH/USDT | 1d | ATR Distance Revert | fixed_atr | high | 0.000000 |
| ETH/USDT | 1d | ATR Distance Revert | fixed_atr | low | 0.000000 |
| ETH/USDT | 1d | ATR Distance Revert | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1d | ATR Distance Revert | structure_trailing | high | 0.000000 |
| ETH/USDT | 1d | ATR Distance Revert | structure_trailing | low | 0.000000 |
| ETH/USDT | 1d | ATR Distance Revert | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1d | ATR Distance Revert | time_based | high | 0.000000 |
| ETH/USDT | 1d | ATR Distance Revert | time_based | low | 0.000000 |
| ETH/USDT | 1d | ATR Distance Revert | time_based | realistic | 0.000000 |
| ETH/USDT | 1d | Bollinger Mean Reversion | fixed_atr | high | 0.000000 |
| ETH/USDT | 1d | Bollinger Mean Reversion | fixed_atr | low | 0.000000 |
| ETH/USDT | 1d | Bollinger Mean Reversion | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1d | Bollinger Mean Reversion | structure_trailing | high | 0.000000 |
| ETH/USDT | 1d | Bollinger Mean Reversion | structure_trailing | low | 0.000000 |
| ETH/USDT | 1d | Bollinger Mean Reversion | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1d | Bollinger Mean Reversion | time_based | high | 0.000000 |
| ETH/USDT | 1d | Bollinger Mean Reversion | time_based | low | 0.000000 |
| ETH/USDT | 1d | Bollinger Mean Reversion | time_based | realistic | 0.000000 |
| ETH/USDT | 1d | Bollinger SnapBack | fixed_atr | high | 0.000000 |
| ETH/USDT | 1d | Bollinger SnapBack | fixed_atr | low | 0.000000 |
| ETH/USDT | 1d | Bollinger SnapBack | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1d | Bollinger SnapBack | structure_trailing | high | 0.000000 |
| ETH/USDT | 1d | Bollinger SnapBack | structure_trailing | low | 0.000000 |
| ETH/USDT | 1d | Bollinger SnapBack | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1d | Bollinger SnapBack | time_based | high | 0.000000 |
| ETH/USDT | 1d | Bollinger SnapBack | time_based | low | 0.000000 |
| ETH/USDT | 1d | Bollinger SnapBack | time_based | realistic | 0.000000 |
| ETH/USDT | 1d | Breakout Retest | fixed_atr | high | 0.000000 |
| ETH/USDT | 1d | Breakout Retest | fixed_atr | low | 0.000000 |
| ETH/USDT | 1d | Breakout Retest | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1d | Breakout Retest | structure_trailing | high | 0.000000 |
| ETH/USDT | 1d | Breakout Retest | structure_trailing | low | 0.000000 |
| ETH/USDT | 1d | Breakout Retest | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1d | Breakout Retest | time_based | high | 0.000000 |
| ETH/USDT | 1d | Breakout Retest | time_based | low | 0.000000 |
| ETH/USDT | 1d | Breakout Retest | time_based | realistic | 0.000000 |
| ETH/USDT | 1d | Donchian Breakout | fixed_atr | high | 0.000000 |
| ETH/USDT | 1d | Donchian Breakout | fixed_atr | low | 0.000000 |
| ETH/USDT | 1d | Donchian Breakout | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1d | Donchian Breakout | structure_trailing | high | 0.000000 |
| ETH/USDT | 1d | Donchian Breakout | structure_trailing | low | 0.000000 |
| ETH/USDT | 1d | Donchian Breakout | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1d | Donchian Breakout | time_based | high | 0.000000 |
| ETH/USDT | 1d | Donchian Breakout | time_based | low | 0.000000 |
| ETH/USDT | 1d | Donchian Breakout | time_based | realistic | 0.000000 |
| ETH/USDT | 1d | MA SlopePullback | fixed_atr | high | 0.000000 |
| ETH/USDT | 1d | MA SlopePullback | fixed_atr | low | 0.000000 |
| ETH/USDT | 1d | MA SlopePullback | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1d | MA SlopePullback | structure_trailing | high | 0.000000 |
| ETH/USDT | 1d | MA SlopePullback | structure_trailing | low | 0.000000 |
| ETH/USDT | 1d | MA SlopePullback | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1d | MA SlopePullback | time_based | high | 0.000000 |
| ETH/USDT | 1d | MA SlopePullback | time_based | low | 0.000000 |
| ETH/USDT | 1d | MA SlopePullback | time_based | realistic | 0.000000 |
| ETH/USDT | 1d | RSI Mean Reversion | fixed_atr | high | 0.000000 |
| ETH/USDT | 1d | RSI Mean Reversion | fixed_atr | low | 0.000000 |
| ETH/USDT | 1d | RSI Mean Reversion | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1d | RSI Mean Reversion | structure_trailing | high | 0.000000 |
| ETH/USDT | 1d | RSI Mean Reversion | structure_trailing | low | 0.000000 |
| ETH/USDT | 1d | RSI Mean Reversion | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1d | RSI Mean Reversion | time_based | high | 0.000000 |
| ETH/USDT | 1d | RSI Mean Reversion | time_based | low | 0.000000 |
| ETH/USDT | 1d | RSI Mean Reversion | time_based | realistic | 0.000000 |
| ETH/USDT | 1d | Range Breakout w/ EMA Trend Filter | fixed_atr | high | 0.000000 |
| ETH/USDT | 1d | Range Breakout w/ EMA Trend Filter | fixed_atr | low | 0.000000 |
| ETH/USDT | 1d | Range Breakout w/ EMA Trend Filter | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1d | Range Breakout w/ EMA Trend Filter | structure_trailing | high | 0.000000 |
| ETH/USDT | 1d | Range Breakout w/ EMA Trend Filter | structure_trailing | low | 0.000000 |
| ETH/USDT | 1d | Range Breakout w/ EMA Trend Filter | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1d | Range Breakout w/ EMA Trend Filter | time_based | high | 0.000000 |
| ETH/USDT | 1d | Range Breakout w/ EMA Trend Filter | time_based | low | 0.000000 |
| ETH/USDT | 1d | Range Breakout w/ EMA Trend Filter | time_based | realistic | 0.000000 |
| ETH/USDT | 1d | Range Fade | fixed_atr | high | 0.000000 |
| ETH/USDT | 1d | Range Fade | fixed_atr | low | 0.000000 |
| ETH/USDT | 1d | Range Fade | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1d | Range Fade | structure_trailing | high | 0.000000 |
| ETH/USDT | 1d | Range Fade | structure_trailing | low | 0.000000 |
| ETH/USDT | 1d | Range Fade | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1d | Range Fade | time_based | high | 0.000000 |
| ETH/USDT | 1d | Range Fade | time_based | low | 0.000000 |
| ETH/USDT | 1d | Range Fade | time_based | realistic | 0.000000 |
| ETH/USDT | 1d | Trend Pullback | fixed_atr | high | 0.000000 |
| ETH/USDT | 1d | Trend Pullback | fixed_atr | low | 0.000000 |
| ETH/USDT | 1d | Trend Pullback | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1d | Trend Pullback | structure_trailing | high | 0.000000 |
| ETH/USDT | 1d | Trend Pullback | structure_trailing | low | 0.000000 |
| ETH/USDT | 1d | Trend Pullback | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1d | Trend Pullback | time_based | high | 0.000000 |
| ETH/USDT | 1d | Trend Pullback | time_based | low | 0.000000 |
| ETH/USDT | 1d | Trend Pullback | time_based | realistic | 0.000000 |
| ETH/USDT | 1d | Vol Compression Breakout | fixed_atr | high | 0.000000 |
| ETH/USDT | 1d | Vol Compression Breakout | fixed_atr | low | 0.000000 |
| ETH/USDT | 1d | Vol Compression Breakout | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1d | Vol Compression Breakout | structure_trailing | high | 0.000000 |
| ETH/USDT | 1d | Vol Compression Breakout | structure_trailing | low | 0.000000 |
| ETH/USDT | 1d | Vol Compression Breakout | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1d | Vol Compression Breakout | time_based | high | 0.000000 |
| ETH/USDT | 1d | Vol Compression Breakout | time_based | low | 0.000000 |
| ETH/USDT | 1d | Vol Compression Breakout | time_based | realistic | 0.000000 |
| ETH/USDT | 1h | ATR Distance Revert | fixed_atr | high | 0.000000 |
| ETH/USDT | 1h | ATR Distance Revert | fixed_atr | low | 0.000000 |
| ETH/USDT | 1h | ATR Distance Revert | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1h | ATR Distance Revert | structure_trailing | high | 0.000000 |
| ETH/USDT | 1h | ATR Distance Revert | structure_trailing | low | 0.000000 |
| ETH/USDT | 1h | ATR Distance Revert | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1h | ATR Distance Revert | time_based | high | 0.000000 |
| ETH/USDT | 1h | ATR Distance Revert | time_based | low | 0.000000 |
| ETH/USDT | 1h | ATR Distance Revert | time_based | realistic | 0.000000 |
| ETH/USDT | 1h | Bollinger Mean Reversion | fixed_atr | high | -56.701257 |
| ETH/USDT | 1h | Bollinger Mean Reversion | fixed_atr | low | -4.671377 |
| ETH/USDT | 1h | Bollinger Mean Reversion | fixed_atr | realistic | -4.671377 |
| ETH/USDT | 1h | Bollinger Mean Reversion | structure_trailing | high | -119.230219 |
| ETH/USDT | 1h | Bollinger Mean Reversion | structure_trailing | low | 18.570555 |
| ETH/USDT | 1h | Bollinger Mean Reversion | structure_trailing | realistic | 18.570555 |
| ETH/USDT | 1h | Bollinger Mean Reversion | time_based | high | 213.366354 |
| ETH/USDT | 1h | Bollinger Mean Reversion | time_based | low | 58.923883 |
| ETH/USDT | 1h | Bollinger Mean Reversion | time_based | realistic | 58.923883 |
| ETH/USDT | 1h | Bollinger SnapBack | fixed_atr | high | -56.701257 |
| ETH/USDT | 1h | Bollinger SnapBack | fixed_atr | low | -4.671377 |
| ETH/USDT | 1h | Bollinger SnapBack | fixed_atr | realistic | -4.671377 |
| ETH/USDT | 1h | Bollinger SnapBack | structure_trailing | high | -119.230219 |
| ETH/USDT | 1h | Bollinger SnapBack | structure_trailing | low | 18.570555 |
| ETH/USDT | 1h | Bollinger SnapBack | structure_trailing | realistic | 18.570555 |
| ETH/USDT | 1h | Bollinger SnapBack | time_based | high | 213.366354 |
| ETH/USDT | 1h | Bollinger SnapBack | time_based | low | 58.923883 |
| ETH/USDT | 1h | Bollinger SnapBack | time_based | realistic | 58.923883 |
| ETH/USDT | 1h | Breakout Retest | fixed_atr | high | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | fixed_atr | low | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | structure_trailing | high | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | structure_trailing | low | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | time_based | high | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | time_based | low | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | time_based | realistic | 0.000000 |
| ETH/USDT | 1h | Donchian Breakout | fixed_atr | high | -422.853696 |
| ETH/USDT | 1h | Donchian Breakout | fixed_atr | low | -335.645648 |
| ETH/USDT | 1h | Donchian Breakout | fixed_atr | realistic | -335.645648 |
| ETH/USDT | 1h | Donchian Breakout | structure_trailing | high | -178.220120 |
| ETH/USDT | 1h | Donchian Breakout | structure_trailing | low | -198.166462 |
| ETH/USDT | 1h | Donchian Breakout | structure_trailing | realistic | -198.166462 |
| ETH/USDT | 1h | Donchian Breakout | time_based | high | -306.428965 |
| ETH/USDT | 1h | Donchian Breakout | time_based | low | -337.617702 |
| ETH/USDT | 1h | Donchian Breakout | time_based | realistic | -337.617702 |
| ETH/USDT | 1h | MA SlopePullback | fixed_atr | high | -259.691800 |
| ETH/USDT | 1h | MA SlopePullback | fixed_atr | low | -282.993048 |
| ETH/USDT | 1h | MA SlopePullback | fixed_atr | realistic | -282.993048 |
| ETH/USDT | 1h | MA SlopePullback | structure_trailing | high | 28.170770 |
| ETH/USDT | 1h | MA SlopePullback | structure_trailing | low | 33.157672 |
| ETH/USDT | 1h | MA SlopePullback | structure_trailing | realistic | 33.157672 |
| ETH/USDT | 1h | MA SlopePullback | time_based | high | -8.076900 |
| ETH/USDT | 1h | MA SlopePullback | time_based | low | -8.076900 |
| ETH/USDT | 1h | MA SlopePullback | time_based | realistic | -8.076900 |
| ETH/USDT | 1h | RSI Mean Reversion | fixed_atr | high | -304.888067 |
| ETH/USDT | 1h | RSI Mean Reversion | fixed_atr | low | -332.875576 |
| ETH/USDT | 1h | RSI Mean Reversion | fixed_atr | realistic | -332.875576 |
| ETH/USDT | 1h | RSI Mean Reversion | structure_trailing | high | -119.309853 |
| ETH/USDT | 1h | RSI Mean Reversion | structure_trailing | low | -204.143872 |
| ETH/USDT | 1h | RSI Mean Reversion | structure_trailing | realistic | -204.143872 |
| ETH/USDT | 1h | RSI Mean Reversion | time_based | high | -188.345321 |
| ETH/USDT | 1h | RSI Mean Reversion | time_based | low | -118.693540 |
| ETH/USDT | 1h | RSI Mean Reversion | time_based | realistic | -118.693540 |
| ETH/USDT | 1h | Range Breakout w/ EMA Trend Filter | fixed_atr | high | 0.000000 |
| ETH/USDT | 1h | Range Breakout w/ EMA Trend Filter | fixed_atr | low | 0.000000 |
| ETH/USDT | 1h | Range Breakout w/ EMA Trend Filter | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1h | Range Breakout w/ EMA Trend Filter | structure_trailing | high | 0.000000 |
| ETH/USDT | 1h | Range Breakout w/ EMA Trend Filter | structure_trailing | low | 0.000000 |
| ETH/USDT | 1h | Range Breakout w/ EMA Trend Filter | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1h | Range Breakout w/ EMA Trend Filter | time_based | high | 0.000000 |
| ETH/USDT | 1h | Range Breakout w/ EMA Trend Filter | time_based | low | 0.000000 |
| ETH/USDT | 1h | Range Breakout w/ EMA Trend Filter | time_based | realistic | 0.000000 |
| ETH/USDT | 1h | Range Fade | fixed_atr | high | 0.000000 |
| ETH/USDT | 1h | Range Fade | fixed_atr | low | 0.000000 |
| ETH/USDT | 1h | Range Fade | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1h | Range Fade | structure_trailing | high | 0.000000 |
| ETH/USDT | 1h | Range Fade | structure_trailing | low | 0.000000 |
| ETH/USDT | 1h | Range Fade | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1h | Range Fade | time_based | high | 0.000000 |
| ETH/USDT | 1h | Range Fade | time_based | low | 0.000000 |
| ETH/USDT | 1h | Range Fade | time_based | realistic | 0.000000 |
| ETH/USDT | 1h | Trend Pullback | fixed_atr | high | 0.000000 |
| ETH/USDT | 1h | Trend Pullback | fixed_atr | low | 0.000000 |
| ETH/USDT | 1h | Trend Pullback | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1h | Trend Pullback | structure_trailing | high | 0.000000 |
| ETH/USDT | 1h | Trend Pullback | structure_trailing | low | 0.000000 |
| ETH/USDT | 1h | Trend Pullback | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1h | Trend Pullback | time_based | high | 0.000000 |
| ETH/USDT | 1h | Trend Pullback | time_based | low | 0.000000 |
| ETH/USDT | 1h | Trend Pullback | time_based | realistic | 0.000000 |
| ETH/USDT | 1h | Vol Compression Breakout | fixed_atr | high | 0.000000 |
| ETH/USDT | 1h | Vol Compression Breakout | fixed_atr | low | 0.000000 |
| ETH/USDT | 1h | Vol Compression Breakout | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1h | Vol Compression Breakout | structure_trailing | high | 0.000000 |
| ETH/USDT | 1h | Vol Compression Breakout | structure_trailing | low | 0.000000 |
| ETH/USDT | 1h | Vol Compression Breakout | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1h | Vol Compression Breakout | time_based | high | 0.000000 |
| ETH/USDT | 1h | Vol Compression Breakout | time_based | low | 0.000000 |
| ETH/USDT | 1h | Vol Compression Breakout | time_based | realistic | 0.000000 |
| ETH/USDT | 2h | ATR Distance Revert | fixed_atr | high | 0.000000 |
| ETH/USDT | 2h | ATR Distance Revert | fixed_atr | low | 0.000000 |
| ETH/USDT | 2h | ATR Distance Revert | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 2h | ATR Distance Revert | structure_trailing | high | 0.000000 |
| ETH/USDT | 2h | ATR Distance Revert | structure_trailing | low | 0.000000 |
| ETH/USDT | 2h | ATR Distance Revert | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 2h | ATR Distance Revert | time_based | high | 0.000000 |
| ETH/USDT | 2h | ATR Distance Revert | time_based | low | 0.000000 |
| ETH/USDT | 2h | ATR Distance Revert | time_based | realistic | 0.000000 |
| ETH/USDT | 2h | Bollinger Mean Reversion | fixed_atr | high | 0.000000 |
| ETH/USDT | 2h | Bollinger Mean Reversion | fixed_atr | low | 0.000000 |
| ETH/USDT | 2h | Bollinger Mean Reversion | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 2h | Bollinger Mean Reversion | structure_trailing | high | 0.000000 |
| ETH/USDT | 2h | Bollinger Mean Reversion | structure_trailing | low | 0.000000 |
| ETH/USDT | 2h | Bollinger Mean Reversion | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 2h | Bollinger Mean Reversion | time_based | high | 0.000000 |
| ETH/USDT | 2h | Bollinger Mean Reversion | time_based | low | 0.000000 |
| ETH/USDT | 2h | Bollinger Mean Reversion | time_based | realistic | 0.000000 |
| ETH/USDT | 2h | Bollinger SnapBack | fixed_atr | high | 0.000000 |
| ETH/USDT | 2h | Bollinger SnapBack | fixed_atr | low | 0.000000 |
| ETH/USDT | 2h | Bollinger SnapBack | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 2h | Bollinger SnapBack | structure_trailing | high | 0.000000 |
| ETH/USDT | 2h | Bollinger SnapBack | structure_trailing | low | 0.000000 |
| ETH/USDT | 2h | Bollinger SnapBack | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 2h | Bollinger SnapBack | time_based | high | 0.000000 |
| ETH/USDT | 2h | Bollinger SnapBack | time_based | low | 0.000000 |
| ETH/USDT | 2h | Bollinger SnapBack | time_based | realistic | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | fixed_atr | high | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | fixed_atr | low | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | structure_trailing | high | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | structure_trailing | low | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | time_based | high | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | time_based | low | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | time_based | realistic | 0.000000 |
| ETH/USDT | 2h | Donchian Breakout | fixed_atr | high | 0.000000 |
| ETH/USDT | 2h | Donchian Breakout | fixed_atr | low | 0.000000 |
| ETH/USDT | 2h | Donchian Breakout | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 2h | Donchian Breakout | structure_trailing | high | 0.000000 |
| ETH/USDT | 2h | Donchian Breakout | structure_trailing | low | 0.000000 |
| ETH/USDT | 2h | Donchian Breakout | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 2h | Donchian Breakout | time_based | high | 0.000000 |
| ETH/USDT | 2h | Donchian Breakout | time_based | low | 0.000000 |
| ETH/USDT | 2h | Donchian Breakout | time_based | realistic | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | fixed_atr | high | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | fixed_atr | low | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | structure_trailing | high | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | structure_trailing | low | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | time_based | high | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | time_based | low | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | time_based | realistic | 0.000000 |
| ETH/USDT | 2h | RSI Mean Reversion | fixed_atr | high | 0.000000 |
| ETH/USDT | 2h | RSI Mean Reversion | fixed_atr | low | 0.000000 |
| ETH/USDT | 2h | RSI Mean Reversion | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 2h | RSI Mean Reversion | structure_trailing | high | 0.000000 |
| ETH/USDT | 2h | RSI Mean Reversion | structure_trailing | low | 0.000000 |
| ETH/USDT | 2h | RSI Mean Reversion | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 2h | RSI Mean Reversion | time_based | high | 0.000000 |
| ETH/USDT | 2h | RSI Mean Reversion | time_based | low | 0.000000 |
| ETH/USDT | 2h | RSI Mean Reversion | time_based | realistic | 0.000000 |
| ETH/USDT | 2h | Range Breakout w/ EMA Trend Filter | fixed_atr | high | 0.000000 |
| ETH/USDT | 2h | Range Breakout w/ EMA Trend Filter | fixed_atr | low | 0.000000 |
| ETH/USDT | 2h | Range Breakout w/ EMA Trend Filter | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 2h | Range Breakout w/ EMA Trend Filter | structure_trailing | high | 0.000000 |
| ETH/USDT | 2h | Range Breakout w/ EMA Trend Filter | structure_trailing | low | 0.000000 |
| ETH/USDT | 2h | Range Breakout w/ EMA Trend Filter | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 2h | Range Breakout w/ EMA Trend Filter | time_based | high | 0.000000 |
| ETH/USDT | 2h | Range Breakout w/ EMA Trend Filter | time_based | low | 0.000000 |
| ETH/USDT | 2h | Range Breakout w/ EMA Trend Filter | time_based | realistic | 0.000000 |
| ETH/USDT | 2h | Range Fade | fixed_atr | high | 0.000000 |
| ETH/USDT | 2h | Range Fade | fixed_atr | low | 0.000000 |
| ETH/USDT | 2h | Range Fade | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 2h | Range Fade | structure_trailing | high | 0.000000 |
| ETH/USDT | 2h | Range Fade | structure_trailing | low | 0.000000 |
| ETH/USDT | 2h | Range Fade | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 2h | Range Fade | time_based | high | 0.000000 |
| ETH/USDT | 2h | Range Fade | time_based | low | 0.000000 |
| ETH/USDT | 2h | Range Fade | time_based | realistic | 0.000000 |
| ETH/USDT | 2h | Trend Pullback | fixed_atr | high | 0.000000 |
| ETH/USDT | 2h | Trend Pullback | fixed_atr | low | 0.000000 |
| ETH/USDT | 2h | Trend Pullback | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 2h | Trend Pullback | structure_trailing | high | 0.000000 |
| ETH/USDT | 2h | Trend Pullback | structure_trailing | low | 0.000000 |
| ETH/USDT | 2h | Trend Pullback | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 2h | Trend Pullback | time_based | high | 0.000000 |
| ETH/USDT | 2h | Trend Pullback | time_based | low | 0.000000 |
| ETH/USDT | 2h | Trend Pullback | time_based | realistic | 0.000000 |
| ETH/USDT | 2h | Vol Compression Breakout | fixed_atr | high | 0.000000 |
| ETH/USDT | 2h | Vol Compression Breakout | fixed_atr | low | 0.000000 |
| ETH/USDT | 2h | Vol Compression Breakout | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 2h | Vol Compression Breakout | structure_trailing | high | 0.000000 |
| ETH/USDT | 2h | Vol Compression Breakout | structure_trailing | low | 0.000000 |
| ETH/USDT | 2h | Vol Compression Breakout | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 2h | Vol Compression Breakout | time_based | high | 0.000000 |
| ETH/USDT | 2h | Vol Compression Breakout | time_based | low | 0.000000 |
| ETH/USDT | 2h | Vol Compression Breakout | time_based | realistic | 0.000000 |
| ETH/USDT | 30m | ATR Distance Revert | fixed_atr | high | 0.000000 |
| ETH/USDT | 30m | ATR Distance Revert | fixed_atr | low | 0.000000 |
| ETH/USDT | 30m | ATR Distance Revert | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 30m | ATR Distance Revert | structure_trailing | high | 0.000000 |
| ETH/USDT | 30m | ATR Distance Revert | structure_trailing | low | 0.000000 |
| ETH/USDT | 30m | ATR Distance Revert | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 30m | ATR Distance Revert | time_based | high | 0.000000 |
| ETH/USDT | 30m | ATR Distance Revert | time_based | low | 0.000000 |
| ETH/USDT | 30m | ATR Distance Revert | time_based | realistic | 0.000000 |
| ETH/USDT | 30m | Bollinger Mean Reversion | fixed_atr | high | -182.758745 |
| ETH/USDT | 30m | Bollinger Mean Reversion | fixed_atr | low | -160.524299 |
| ETH/USDT | 30m | Bollinger Mean Reversion | fixed_atr | realistic | -160.524299 |
| ETH/USDT | 30m | Bollinger Mean Reversion | structure_trailing | high | 119.595196 |
| ETH/USDT | 30m | Bollinger Mean Reversion | structure_trailing | low | 78.708938 |
| ETH/USDT | 30m | Bollinger Mean Reversion | structure_trailing | realistic | 78.708938 |
| ETH/USDT | 30m | Bollinger Mean Reversion | time_based | high | -114.895945 |
| ETH/USDT | 30m | Bollinger Mean Reversion | time_based | low | -92.872957 |
| ETH/USDT | 30m | Bollinger Mean Reversion | time_based | realistic | -92.872957 |
| ETH/USDT | 30m | Bollinger SnapBack | fixed_atr | high | -182.758745 |
| ETH/USDT | 30m | Bollinger SnapBack | fixed_atr | low | -160.524299 |
| ETH/USDT | 30m | Bollinger SnapBack | fixed_atr | realistic | -160.524299 |
| ETH/USDT | 30m | Bollinger SnapBack | structure_trailing | high | 119.595196 |
| ETH/USDT | 30m | Bollinger SnapBack | structure_trailing | low | 78.708938 |
| ETH/USDT | 30m | Bollinger SnapBack | structure_trailing | realistic | 78.708938 |
| ETH/USDT | 30m | Bollinger SnapBack | time_based | high | -114.895945 |
| ETH/USDT | 30m | Bollinger SnapBack | time_based | low | -92.872957 |
| ETH/USDT | 30m | Bollinger SnapBack | time_based | realistic | -92.872957 |
| ETH/USDT | 30m | Breakout Retest | fixed_atr | high | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | fixed_atr | low | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | structure_trailing | high | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | structure_trailing | low | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | time_based | high | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | time_based | low | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | time_based | realistic | 0.000000 |
| ETH/USDT | 30m | Donchian Breakout | fixed_atr | high | -253.911784 |
| ETH/USDT | 30m | Donchian Breakout | fixed_atr | low | -239.079092 |
| ETH/USDT | 30m | Donchian Breakout | fixed_atr | realistic | -239.079092 |
| ETH/USDT | 30m | Donchian Breakout | structure_trailing | high | -47.262264 |
| ETH/USDT | 30m | Donchian Breakout | structure_trailing | low | -170.688827 |
| ETH/USDT | 30m | Donchian Breakout | structure_trailing | realistic | -170.688827 |
| ETH/USDT | 30m | Donchian Breakout | time_based | high | -419.536807 |
| ETH/USDT | 30m | Donchian Breakout | time_based | low | -563.940423 |
| ETH/USDT | 30m | Donchian Breakout | time_based | realistic | -563.940423 |
| ETH/USDT | 30m | MA SlopePullback | fixed_atr | high | 40.274684 |
| ETH/USDT | 30m | MA SlopePullback | fixed_atr | low | 54.262318 |
| ETH/USDT | 30m | MA SlopePullback | fixed_atr | realistic | 54.262318 |
| ETH/USDT | 30m | MA SlopePullback | structure_trailing | high | 121.903200 |
| ETH/USDT | 30m | MA SlopePullback | structure_trailing | low | -21.593602 |
| ETH/USDT | 30m | MA SlopePullback | structure_trailing | realistic | -21.593602 |
| ETH/USDT | 30m | MA SlopePullback | time_based | high | 11.044237 |
| ETH/USDT | 30m | MA SlopePullback | time_based | low | 21.266809 |
| ETH/USDT | 30m | MA SlopePullback | time_based | realistic | 21.266809 |
| ETH/USDT | 30m | RSI Mean Reversion | fixed_atr | high | 0.000000 |
| ETH/USDT | 30m | RSI Mean Reversion | fixed_atr | low | 0.000000 |
| ETH/USDT | 30m | RSI Mean Reversion | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 30m | RSI Mean Reversion | structure_trailing | high | 0.000000 |
| ETH/USDT | 30m | RSI Mean Reversion | structure_trailing | low | 0.000000 |
| ETH/USDT | 30m | RSI Mean Reversion | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 30m | RSI Mean Reversion | time_based | high | 0.000000 |
| ETH/USDT | 30m | RSI Mean Reversion | time_based | low | 0.000000 |
| ETH/USDT | 30m | RSI Mean Reversion | time_based | realistic | 0.000000 |
| ETH/USDT | 30m | Range Breakout w/ EMA Trend Filter | fixed_atr | high | 0.000000 |
| ETH/USDT | 30m | Range Breakout w/ EMA Trend Filter | fixed_atr | low | 0.000000 |
| ETH/USDT | 30m | Range Breakout w/ EMA Trend Filter | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 30m | Range Breakout w/ EMA Trend Filter | structure_trailing | high | 0.000000 |
| ETH/USDT | 30m | Range Breakout w/ EMA Trend Filter | structure_trailing | low | 0.000000 |
| ETH/USDT | 30m | Range Breakout w/ EMA Trend Filter | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 30m | Range Breakout w/ EMA Trend Filter | time_based | high | 0.000000 |
| ETH/USDT | 30m | Range Breakout w/ EMA Trend Filter | time_based | low | 0.000000 |
| ETH/USDT | 30m | Range Breakout w/ EMA Trend Filter | time_based | realistic | 0.000000 |
| ETH/USDT | 30m | Range Fade | fixed_atr | high | 0.000000 |
| ETH/USDT | 30m | Range Fade | fixed_atr | low | 0.000000 |
| ETH/USDT | 30m | Range Fade | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 30m | Range Fade | structure_trailing | high | 0.000000 |
| ETH/USDT | 30m | Range Fade | structure_trailing | low | 0.000000 |
| ETH/USDT | 30m | Range Fade | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 30m | Range Fade | time_based | high | 0.000000 |
| ETH/USDT | 30m | Range Fade | time_based | low | 0.000000 |
| ETH/USDT | 30m | Range Fade | time_based | realistic | 0.000000 |
| ETH/USDT | 30m | Trend Pullback | fixed_atr | high | 0.000000 |
| ETH/USDT | 30m | Trend Pullback | fixed_atr | low | 0.000000 |
| ETH/USDT | 30m | Trend Pullback | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 30m | Trend Pullback | structure_trailing | high | 0.000000 |
| ETH/USDT | 30m | Trend Pullback | structure_trailing | low | 0.000000 |
| ETH/USDT | 30m | Trend Pullback | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 30m | Trend Pullback | time_based | high | 0.000000 |
| ETH/USDT | 30m | Trend Pullback | time_based | low | 0.000000 |
| ETH/USDT | 30m | Trend Pullback | time_based | realistic | 0.000000 |
| ETH/USDT | 30m | Vol Compression Breakout | fixed_atr | high | 0.000000 |
| ETH/USDT | 30m | Vol Compression Breakout | fixed_atr | low | 0.000000 |
| ETH/USDT | 30m | Vol Compression Breakout | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 30m | Vol Compression Breakout | structure_trailing | high | 0.000000 |
| ETH/USDT | 30m | Vol Compression Breakout | structure_trailing | low | 0.000000 |
| ETH/USDT | 30m | Vol Compression Breakout | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 30m | Vol Compression Breakout | time_based | high | 0.000000 |
| ETH/USDT | 30m | Vol Compression Breakout | time_based | low | 0.000000 |
| ETH/USDT | 30m | Vol Compression Breakout | time_based | realistic | 0.000000 |
| ETH/USDT | 4h | ATR Distance Revert | fixed_atr | high | 0.000000 |
| ETH/USDT | 4h | ATR Distance Revert | fixed_atr | low | 0.000000 |
| ETH/USDT | 4h | ATR Distance Revert | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 4h | ATR Distance Revert | structure_trailing | high | 0.000000 |
| ETH/USDT | 4h | ATR Distance Revert | structure_trailing | low | 0.000000 |
| ETH/USDT | 4h | ATR Distance Revert | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 4h | ATR Distance Revert | time_based | high | 0.000000 |
| ETH/USDT | 4h | ATR Distance Revert | time_based | low | 0.000000 |
| ETH/USDT | 4h | ATR Distance Revert | time_based | realistic | 0.000000 |
| ETH/USDT | 4h | Bollinger Mean Reversion | fixed_atr | high | 0.000000 |
| ETH/USDT | 4h | Bollinger Mean Reversion | fixed_atr | low | 0.000000 |
| ETH/USDT | 4h | Bollinger Mean Reversion | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 4h | Bollinger Mean Reversion | structure_trailing | high | 0.000000 |
| ETH/USDT | 4h | Bollinger Mean Reversion | structure_trailing | low | 0.000000 |
| ETH/USDT | 4h | Bollinger Mean Reversion | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 4h | Bollinger Mean Reversion | time_based | high | 0.000000 |
| ETH/USDT | 4h | Bollinger Mean Reversion | time_based | low | 0.000000 |
| ETH/USDT | 4h | Bollinger Mean Reversion | time_based | realistic | 0.000000 |
| ETH/USDT | 4h | Bollinger SnapBack | fixed_atr | high | 0.000000 |
| ETH/USDT | 4h | Bollinger SnapBack | fixed_atr | low | 0.000000 |
| ETH/USDT | 4h | Bollinger SnapBack | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 4h | Bollinger SnapBack | structure_trailing | high | 0.000000 |
| ETH/USDT | 4h | Bollinger SnapBack | structure_trailing | low | 0.000000 |
| ETH/USDT | 4h | Bollinger SnapBack | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 4h | Bollinger SnapBack | time_based | high | 0.000000 |
| ETH/USDT | 4h | Bollinger SnapBack | time_based | low | 0.000000 |
| ETH/USDT | 4h | Bollinger SnapBack | time_based | realistic | 0.000000 |
| ETH/USDT | 4h | Breakout Retest | fixed_atr | high | 0.000000 |
| ETH/USDT | 4h | Breakout Retest | fixed_atr | low | 0.000000 |
| ETH/USDT | 4h | Breakout Retest | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 4h | Breakout Retest | structure_trailing | high | 0.000000 |
| ETH/USDT | 4h | Breakout Retest | structure_trailing | low | 0.000000 |
| ETH/USDT | 4h | Breakout Retest | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 4h | Breakout Retest | time_based | high | 0.000000 |
| ETH/USDT | 4h | Breakout Retest | time_based | low | 0.000000 |
| ETH/USDT | 4h | Breakout Retest | time_based | realistic | 0.000000 |
| ETH/USDT | 4h | Donchian Breakout | fixed_atr | high | 0.000000 |
| ETH/USDT | 4h | Donchian Breakout | fixed_atr | low | 0.000000 |
| ETH/USDT | 4h | Donchian Breakout | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 4h | Donchian Breakout | structure_trailing | high | 0.000000 |
| ETH/USDT | 4h | Donchian Breakout | structure_trailing | low | 0.000000 |
| ETH/USDT | 4h | Donchian Breakout | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 4h | Donchian Breakout | time_based | high | 0.000000 |
| ETH/USDT | 4h | Donchian Breakout | time_based | low | 0.000000 |
| ETH/USDT | 4h | Donchian Breakout | time_based | realistic | 0.000000 |
| ETH/USDT | 4h | MA SlopePullback | fixed_atr | high | 0.000000 |
| ETH/USDT | 4h | MA SlopePullback | fixed_atr | low | 0.000000 |
| ETH/USDT | 4h | MA SlopePullback | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 4h | MA SlopePullback | structure_trailing | high | 0.000000 |
| ETH/USDT | 4h | MA SlopePullback | structure_trailing | low | 0.000000 |
| ETH/USDT | 4h | MA SlopePullback | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 4h | MA SlopePullback | time_based | high | 0.000000 |
| ETH/USDT | 4h | MA SlopePullback | time_based | low | 0.000000 |
| ETH/USDT | 4h | MA SlopePullback | time_based | realistic | 0.000000 |
| ETH/USDT | 4h | RSI Mean Reversion | fixed_atr | high | 0.000000 |
| ETH/USDT | 4h | RSI Mean Reversion | fixed_atr | low | 0.000000 |
| ETH/USDT | 4h | RSI Mean Reversion | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 4h | RSI Mean Reversion | structure_trailing | high | 0.000000 |
| ETH/USDT | 4h | RSI Mean Reversion | structure_trailing | low | 0.000000 |
| ETH/USDT | 4h | RSI Mean Reversion | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 4h | RSI Mean Reversion | time_based | high | 0.000000 |
| ETH/USDT | 4h | RSI Mean Reversion | time_based | low | 0.000000 |
| ETH/USDT | 4h | RSI Mean Reversion | time_based | realistic | 0.000000 |
| ETH/USDT | 4h | Range Breakout w/ EMA Trend Filter | fixed_atr | high | 0.000000 |
| ETH/USDT | 4h | Range Breakout w/ EMA Trend Filter | fixed_atr | low | 0.000000 |
| ETH/USDT | 4h | Range Breakout w/ EMA Trend Filter | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 4h | Range Breakout w/ EMA Trend Filter | structure_trailing | high | 0.000000 |
| ETH/USDT | 4h | Range Breakout w/ EMA Trend Filter | structure_trailing | low | 0.000000 |
| ETH/USDT | 4h | Range Breakout w/ EMA Trend Filter | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 4h | Range Breakout w/ EMA Trend Filter | time_based | high | 0.000000 |
| ETH/USDT | 4h | Range Breakout w/ EMA Trend Filter | time_based | low | 0.000000 |
| ETH/USDT | 4h | Range Breakout w/ EMA Trend Filter | time_based | realistic | 0.000000 |
| ETH/USDT | 4h | Range Fade | fixed_atr | high | 0.000000 |
| ETH/USDT | 4h | Range Fade | fixed_atr | low | 0.000000 |
| ETH/USDT | 4h | Range Fade | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 4h | Range Fade | structure_trailing | high | 0.000000 |
| ETH/USDT | 4h | Range Fade | structure_trailing | low | 0.000000 |
| ETH/USDT | 4h | Range Fade | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 4h | Range Fade | time_based | high | 0.000000 |
| ETH/USDT | 4h | Range Fade | time_based | low | 0.000000 |
| ETH/USDT | 4h | Range Fade | time_based | realistic | 0.000000 |
| ETH/USDT | 4h | Trend Pullback | fixed_atr | high | 0.000000 |
| ETH/USDT | 4h | Trend Pullback | fixed_atr | low | 0.000000 |
| ETH/USDT | 4h | Trend Pullback | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 4h | Trend Pullback | structure_trailing | high | 0.000000 |
| ETH/USDT | 4h | Trend Pullback | structure_trailing | low | 0.000000 |
| ETH/USDT | 4h | Trend Pullback | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 4h | Trend Pullback | time_based | high | 0.000000 |
| ETH/USDT | 4h | Trend Pullback | time_based | low | 0.000000 |
| ETH/USDT | 4h | Trend Pullback | time_based | realistic | 0.000000 |
| ETH/USDT | 4h | Vol Compression Breakout | fixed_atr | high | 0.000000 |
| ETH/USDT | 4h | Vol Compression Breakout | fixed_atr | low | 0.000000 |
| ETH/USDT | 4h | Vol Compression Breakout | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 4h | Vol Compression Breakout | structure_trailing | high | 0.000000 |
| ETH/USDT | 4h | Vol Compression Breakout | structure_trailing | low | 0.000000 |
| ETH/USDT | 4h | Vol Compression Breakout | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 4h | Vol Compression Breakout | time_based | high | 0.000000 |
| ETH/USDT | 4h | Vol Compression Breakout | time_based | low | 0.000000 |
| ETH/USDT | 4h | Vol Compression Breakout | time_based | realistic | 0.000000 |

## Monte Carlo Validation Summary (Top 20%)
- no combinations selected for Monte Carlo

## Verdict
- `NO ROBUST EDGE`
