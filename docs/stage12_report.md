# Stage-12 Full Price-Family Robustness Sweep

- run_id: `20260301_041932_2fc2823beb24_stage12`
- symbols: `BTC/USDT, ETH/USDT`
- timeframes: `15m, 30m, 1h, 2h, 4h, 1d`
- total_combinations: `1188`
- valid_combinations: `0`
- runtime_seconds: `19.998`
- verdict: `NO ROBUST EDGE`

## Runtime Breakdown by Timeframe
| timeframe | runtime_seconds |
| --- | ---: |
| 15m | 7.3017 |
| 1d | 1.2325 |
| 1h | 2.7388 |
| 2h | 1.9274 |
| 30m | 4.2936 |
| 4h | 1.4516 |

## Top 10 Robust Combinations
| symbol | timeframe | strategy | exit_type | cost_level | exp_lcb | PF | expectancy | robust_score | cost_sensitivity | stability | MC_p_ruin | MC_p_return_negative |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| BTC/USDT | 2h | Bollinger SnapBack | time_based | low | 940.733404 | 0.000000 | 940.733404 | 836.570024 | 103.163380 | INVALID | nan | nan |
| BTC/USDT | 2h | Bollinger SnapBack | time_based | realistic | 940.733404 | 0.000000 | 940.733404 | 836.570024 | 103.163380 | INVALID | nan | nan |
| BTC/USDT | 2h | Bollinger SnapBack | time_based | high | 734.406644 | 0.000000 | 734.406644 | 836.570024 | 103.163380 | INVALID | nan | nan |
| BTC/USDT | 2h | Bollinger Mean Reversion | time_based | low | 921.700775 | 0.000000 | 921.700775 | 817.716778 | 102.983998 | INVALID | nan | nan |
| BTC/USDT | 2h | Bollinger Mean Reversion | time_based | realistic | 921.700775 | 0.000000 | 921.700775 | 817.716778 | 102.983998 | INVALID | nan | nan |
| BTC/USDT | 2h | Bollinger Mean Reversion | time_based | high | 715.732780 | 0.000000 | 715.732780 | 817.716778 | 102.983998 | INVALID | nan | nan |
| BTC/USDT | 2h | Bollinger Mean Reversion | fixed_atr | low | 896.270925 | 0.000000 | 896.270925 | 805.001853 | 90.269073 | INVALID | nan | nan |
| BTC/USDT | 2h | Bollinger Mean Reversion | fixed_atr | realistic | 896.270925 | 0.000000 | 896.270925 | 805.001853 | 90.269073 | INVALID | nan | nan |
| BTC/USDT | 2h | Bollinger Mean Reversion | fixed_atr | high | 715.732780 | 0.000000 | 715.732780 | 805.001853 | 90.269073 | INVALID | nan | nan |
| BTC/USDT | 2h | Bollinger SnapBack | fixed_atr | low | 865.275879 | 0.000000 | 865.275879 | 798.841262 | 65.434618 | INVALID | nan | nan |

## Per-Timeframe Summary
| timeframe | combos | valid | exp_lcb_median | best_robust_score |
| --- | ---: | ---: | ---: | ---: |
| 15m | 198 | 0 | -54.298422 | 589.202124 |
| 1d | 198 | 0 | 0.000000 | -1.000000 |
| 1h | 198 | 0 | 0.000000 | 658.231691 |
| 2h | 198 | 0 | 0.000000 | 836.570024 |
| 30m | 198 | 0 | 0.000000 | 46.268501 |
| 4h | 198 | 0 | 0.000000 | 395.313484 |

## Per-Symbol Summary
| symbol | combos | valid | exp_lcb_median | best_robust_score |
| --- | ---: | ---: | ---: | ---: |
| BTC/USDT | 594 | 0 | 0.000000 | 836.570024 |
| ETH/USDT | 594 | 0 | 0.000000 | 589.202124 |

## Stability Heatmap Table
| symbol | timeframe | STABLE | UNSTABLE | ZERO_TRADE | INVALID | INSUFFICIENT_DATA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BTC/USDT | 15m | 0 | 0 | 27 | 72 | 0 |
| BTC/USDT | 1d | 0 | 0 | 99 | 0 | 0 |
| BTC/USDT | 1h | 0 | 0 | 54 | 45 | 0 |
| BTC/USDT | 2h | 0 | 0 | 63 | 36 | 0 |
| BTC/USDT | 30m | 0 | 0 | 45 | 54 | 0 |
| BTC/USDT | 4h | 0 | 0 | 99 | 0 | 0 |
| ETH/USDT | 15m | 0 | 0 | 18 | 81 | 0 |
| ETH/USDT | 1d | 0 | 0 | 99 | 0 | 0 |
| ETH/USDT | 1h | 0 | 0 | 45 | 54 | 0 |
| ETH/USDT | 2h | 0 | 0 | 63 | 36 | 0 |
| ETH/USDT | 30m | 0 | 0 | 45 | 54 | 0 |
| ETH/USDT | 4h | 0 | 0 | 90 | 9 | 0 |

## Cost Sensitivity Chart Data
| symbol | timeframe | strategy | exit_type | cost_level | exp_lcb |
| --- | --- | --- | --- | --- | ---: |
| BTC/USDT | 15m | ATR Distance Revert | fixed_atr | high | -71.899968 |
| BTC/USDT | 15m | ATR Distance Revert | fixed_atr | low | -73.272946 |
| BTC/USDT | 15m | ATR Distance Revert | fixed_atr | realistic | -73.272946 |
| BTC/USDT | 15m | ATR Distance Revert | structure_trailing | high | -44.980139 |
| BTC/USDT | 15m | ATR Distance Revert | structure_trailing | low | -56.309890 |
| BTC/USDT | 15m | ATR Distance Revert | structure_trailing | realistic | -56.309890 |
| BTC/USDT | 15m | ATR Distance Revert | time_based | high | -146.369721 |
| BTC/USDT | 15m | ATR Distance Revert | time_based | low | -146.188038 |
| BTC/USDT | 15m | ATR Distance Revert | time_based | realistic | -146.188038 |
| BTC/USDT | 15m | Bollinger Mean Reversion | fixed_atr | high | -41.852179 |
| BTC/USDT | 15m | Bollinger Mean Reversion | fixed_atr | low | -49.157925 |
| BTC/USDT | 15m | Bollinger Mean Reversion | fixed_atr | realistic | -49.157925 |
| BTC/USDT | 15m | Bollinger Mean Reversion | structure_trailing | high | -53.951592 |
| BTC/USDT | 15m | Bollinger Mean Reversion | structure_trailing | low | -63.190000 |
| BTC/USDT | 15m | Bollinger Mean Reversion | structure_trailing | realistic | -63.190000 |
| BTC/USDT | 15m | Bollinger Mean Reversion | time_based | high | -134.632506 |
| BTC/USDT | 15m | Bollinger Mean Reversion | time_based | low | -103.861244 |
| BTC/USDT | 15m | Bollinger Mean Reversion | time_based | realistic | -103.861244 |
| BTC/USDT | 15m | Bollinger SnapBack | fixed_atr | high | -43.043635 |
| BTC/USDT | 15m | Bollinger SnapBack | fixed_atr | low | -57.958879 |
| BTC/USDT | 15m | Bollinger SnapBack | fixed_atr | realistic | -57.958879 |
| BTC/USDT | 15m | Bollinger SnapBack | structure_trailing | high | -58.789875 |
| BTC/USDT | 15m | Bollinger SnapBack | structure_trailing | low | -69.090705 |
| BTC/USDT | 15m | Bollinger SnapBack | structure_trailing | realistic | -69.090705 |
| BTC/USDT | 15m | Bollinger SnapBack | time_based | high | -106.861153 |
| BTC/USDT | 15m | Bollinger SnapBack | time_based | low | -74.673661 |
| BTC/USDT | 15m | Bollinger SnapBack | time_based | realistic | -74.673661 |
| BTC/USDT | 15m | Breakout Retest | fixed_atr | high | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | fixed_atr | low | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | structure_trailing | high | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | structure_trailing | low | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | time_based | high | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | time_based | low | 0.000000 |
| BTC/USDT | 15m | Breakout Retest | time_based | realistic | 0.000000 |
| BTC/USDT | 15m | Donchian Breakout | fixed_atr | high | -50.582273 |
| BTC/USDT | 15m | Donchian Breakout | fixed_atr | low | -56.755400 |
| BTC/USDT | 15m | Donchian Breakout | fixed_atr | realistic | -56.755400 |
| BTC/USDT | 15m | Donchian Breakout | structure_trailing | high | -42.139935 |
| BTC/USDT | 15m | Donchian Breakout | structure_trailing | low | -58.644066 |
| BTC/USDT | 15m | Donchian Breakout | structure_trailing | realistic | -58.644066 |
| BTC/USDT | 15m | Donchian Breakout | time_based | high | -5.006274 |
| BTC/USDT | 15m | Donchian Breakout | time_based | low | -9.033179 |
| BTC/USDT | 15m | Donchian Breakout | time_based | realistic | -9.033179 |
| BTC/USDT | 15m | MA SlopePullback | fixed_atr | high | -6.429647 |
| BTC/USDT | 15m | MA SlopePullback | fixed_atr | low | -7.704601 |
| BTC/USDT | 15m | MA SlopePullback | fixed_atr | realistic | -7.704601 |
| BTC/USDT | 15m | MA SlopePullback | structure_trailing | high | 6.817120 |
| BTC/USDT | 15m | MA SlopePullback | structure_trailing | low | -24.915279 |
| BTC/USDT | 15m | MA SlopePullback | structure_trailing | realistic | -24.915279 |
| BTC/USDT | 15m | MA SlopePullback | time_based | high | -18.560909 |
| BTC/USDT | 15m | MA SlopePullback | time_based | low | 17.962247 |
| BTC/USDT | 15m | MA SlopePullback | time_based | realistic | 17.962247 |
| BTC/USDT | 15m | RSI Mean Reversion | fixed_atr | high | -97.821419 |
| BTC/USDT | 15m | RSI Mean Reversion | fixed_atr | low | -104.139944 |
| BTC/USDT | 15m | RSI Mean Reversion | fixed_atr | realistic | -104.139944 |
| BTC/USDT | 15m | RSI Mean Reversion | structure_trailing | high | -64.324760 |
| BTC/USDT | 15m | RSI Mean Reversion | structure_trailing | low | -70.484856 |
| BTC/USDT | 15m | RSI Mean Reversion | structure_trailing | realistic | -70.484856 |
| BTC/USDT | 15m | RSI Mean Reversion | time_based | high | -222.226120 |
| BTC/USDT | 15m | RSI Mean Reversion | time_based | low | -209.328667 |
| BTC/USDT | 15m | RSI Mean Reversion | time_based | realistic | -209.328667 |
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
| BTC/USDT | 15m | Trend Pullback | fixed_atr | high | -140.966240 |
| BTC/USDT | 15m | Trend Pullback | fixed_atr | low | -180.081160 |
| BTC/USDT | 15m | Trend Pullback | fixed_atr | realistic | -180.081160 |
| BTC/USDT | 15m | Trend Pullback | structure_trailing | high | -133.792313 |
| BTC/USDT | 15m | Trend Pullback | structure_trailing | low | -98.070838 |
| BTC/USDT | 15m | Trend Pullback | structure_trailing | realistic | -98.070838 |
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
| BTC/USDT | 1h | ATR Distance Revert | fixed_atr | high | 95.434290 |
| BTC/USDT | 1h | ATR Distance Revert | fixed_atr | low | 63.536302 |
| BTC/USDT | 1h | ATR Distance Revert | fixed_atr | realistic | 63.536302 |
| BTC/USDT | 1h | ATR Distance Revert | structure_trailing | high | -104.230048 |
| BTC/USDT | 1h | ATR Distance Revert | structure_trailing | low | -138.317098 |
| BTC/USDT | 1h | ATR Distance Revert | structure_trailing | realistic | -138.317098 |
| BTC/USDT | 1h | ATR Distance Revert | time_based | high | 841.421146 |
| BTC/USDT | 1h | ATR Distance Revert | time_based | low | 719.961509 |
| BTC/USDT | 1h | ATR Distance Revert | time_based | realistic | 719.961509 |
| BTC/USDT | 1h | Bollinger Mean Reversion | fixed_atr | high | -264.088666 |
| BTC/USDT | 1h | Bollinger Mean Reversion | fixed_atr | low | -275.856142 |
| BTC/USDT | 1h | Bollinger Mean Reversion | fixed_atr | realistic | -275.856142 |
| BTC/USDT | 1h | Bollinger Mean Reversion | structure_trailing | high | -93.427934 |
| BTC/USDT | 1h | Bollinger Mean Reversion | structure_trailing | low | -142.439466 |
| BTC/USDT | 1h | Bollinger Mean Reversion | structure_trailing | realistic | -142.439466 |
| BTC/USDT | 1h | Bollinger Mean Reversion | time_based | high | 69.544646 |
| BTC/USDT | 1h | Bollinger Mean Reversion | time_based | low | -81.244324 |
| BTC/USDT | 1h | Bollinger Mean Reversion | time_based | realistic | -81.244324 |
| BTC/USDT | 1h | Bollinger SnapBack | fixed_atr | high | -157.698541 |
| BTC/USDT | 1h | Bollinger SnapBack | fixed_atr | low | -168.062065 |
| BTC/USDT | 1h | Bollinger SnapBack | fixed_atr | realistic | -168.062065 |
| BTC/USDT | 1h | Bollinger SnapBack | structure_trailing | high | -68.910010 |
| BTC/USDT | 1h | Bollinger SnapBack | structure_trailing | low | -130.418252 |
| BTC/USDT | 1h | Bollinger SnapBack | structure_trailing | realistic | -130.418252 |
| BTC/USDT | 1h | Bollinger SnapBack | time_based | high | 84.889657 |
| BTC/USDT | 1h | Bollinger SnapBack | time_based | low | -66.255805 |
| BTC/USDT | 1h | Bollinger SnapBack | time_based | realistic | -66.255805 |
| BTC/USDT | 1h | Breakout Retest | fixed_atr | high | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | fixed_atr | low | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | structure_trailing | high | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | structure_trailing | low | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | time_based | high | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | time_based | low | 0.000000 |
| BTC/USDT | 1h | Breakout Retest | time_based | realistic | 0.000000 |
| BTC/USDT | 1h | Donchian Breakout | fixed_atr | high | -254.449830 |
| BTC/USDT | 1h | Donchian Breakout | fixed_atr | low | -271.036625 |
| BTC/USDT | 1h | Donchian Breakout | fixed_atr | realistic | -271.036625 |
| BTC/USDT | 1h | Donchian Breakout | structure_trailing | high | -199.429936 |
| BTC/USDT | 1h | Donchian Breakout | structure_trailing | low | -114.408698 |
| BTC/USDT | 1h | Donchian Breakout | structure_trailing | realistic | -114.408698 |
| BTC/USDT | 1h | Donchian Breakout | time_based | high | -573.310224 |
| BTC/USDT | 1h | Donchian Breakout | time_based | low | -456.505860 |
| BTC/USDT | 1h | Donchian Breakout | time_based | realistic | -456.505860 |
| BTC/USDT | 1h | MA SlopePullback | fixed_atr | high | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | fixed_atr | low | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | structure_trailing | high | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | structure_trailing | low | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | time_based | high | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | time_based | low | 0.000000 |
| BTC/USDT | 1h | MA SlopePullback | time_based | realistic | 0.000000 |
| BTC/USDT | 1h | RSI Mean Reversion | fixed_atr | high | -52.073241 |
| BTC/USDT | 1h | RSI Mean Reversion | fixed_atr | low | -29.220147 |
| BTC/USDT | 1h | RSI Mean Reversion | fixed_atr | realistic | -29.220147 |
| BTC/USDT | 1h | RSI Mean Reversion | structure_trailing | high | -81.758680 |
| BTC/USDT | 1h | RSI Mean Reversion | structure_trailing | low | -109.887229 |
| BTC/USDT | 1h | RSI Mean Reversion | structure_trailing | realistic | -109.887229 |
| BTC/USDT | 1h | RSI Mean Reversion | time_based | high | -410.357611 |
| BTC/USDT | 1h | RSI Mean Reversion | time_based | low | -329.598188 |
| BTC/USDT | 1h | RSI Mean Reversion | time_based | realistic | -329.598188 |
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
| BTC/USDT | 2h | Bollinger Mean Reversion | fixed_atr | high | 715.732780 |
| BTC/USDT | 2h | Bollinger Mean Reversion | fixed_atr | low | 896.270925 |
| BTC/USDT | 2h | Bollinger Mean Reversion | fixed_atr | realistic | 896.270925 |
| BTC/USDT | 2h | Bollinger Mean Reversion | structure_trailing | high | 29.334580 |
| BTC/USDT | 2h | Bollinger Mean Reversion | structure_trailing | low | -131.450769 |
| BTC/USDT | 2h | Bollinger Mean Reversion | structure_trailing | realistic | -131.450769 |
| BTC/USDT | 2h | Bollinger Mean Reversion | time_based | high | 715.732780 |
| BTC/USDT | 2h | Bollinger Mean Reversion | time_based | low | 921.700775 |
| BTC/USDT | 2h | Bollinger Mean Reversion | time_based | realistic | 921.700775 |
| BTC/USDT | 2h | Bollinger SnapBack | fixed_atr | high | 734.406644 |
| BTC/USDT | 2h | Bollinger SnapBack | fixed_atr | low | 865.275879 |
| BTC/USDT | 2h | Bollinger SnapBack | fixed_atr | realistic | 865.275879 |
| BTC/USDT | 2h | Bollinger SnapBack | structure_trailing | high | 393.731195 |
| BTC/USDT | 2h | Bollinger SnapBack | structure_trailing | low | -1.403842 |
| BTC/USDT | 2h | Bollinger SnapBack | structure_trailing | realistic | -1.403842 |
| BTC/USDT | 2h | Bollinger SnapBack | time_based | high | 734.406644 |
| BTC/USDT | 2h | Bollinger SnapBack | time_based | low | 940.733404 |
| BTC/USDT | 2h | Bollinger SnapBack | time_based | realistic | 940.733404 |
| BTC/USDT | 2h | Breakout Retest | fixed_atr | high | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | fixed_atr | low | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | structure_trailing | high | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | structure_trailing | low | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | time_based | high | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | time_based | low | 0.000000 |
| BTC/USDT | 2h | Breakout Retest | time_based | realistic | 0.000000 |
| BTC/USDT | 2h | Donchian Breakout | fixed_atr | high | -455.855067 |
| BTC/USDT | 2h | Donchian Breakout | fixed_atr | low | -464.859160 |
| BTC/USDT | 2h | Donchian Breakout | fixed_atr | realistic | -464.859160 |
| BTC/USDT | 2h | Donchian Breakout | structure_trailing | high | -455.855067 |
| BTC/USDT | 2h | Donchian Breakout | structure_trailing | low | -366.525505 |
| BTC/USDT | 2h | Donchian Breakout | structure_trailing | realistic | -366.525505 |
| BTC/USDT | 2h | Donchian Breakout | time_based | high | -798.238159 |
| BTC/USDT | 2h | Donchian Breakout | time_based | low | -1005.599136 |
| BTC/USDT | 2h | Donchian Breakout | time_based | realistic | -1005.599136 |
| BTC/USDT | 2h | MA SlopePullback | fixed_atr | high | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | fixed_atr | low | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | structure_trailing | high | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | structure_trailing | low | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | time_based | high | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | time_based | low | 0.000000 |
| BTC/USDT | 2h | MA SlopePullback | time_based | realistic | 0.000000 |
| BTC/USDT | 2h | RSI Mean Reversion | fixed_atr | high | -42.533713 |
| BTC/USDT | 2h | RSI Mean Reversion | fixed_atr | low | 232.726550 |
| BTC/USDT | 2h | RSI Mean Reversion | fixed_atr | realistic | 232.726550 |
| BTC/USDT | 2h | RSI Mean Reversion | structure_trailing | high | 42.649944 |
| BTC/USDT | 2h | RSI Mean Reversion | structure_trailing | low | 7.588095 |
| BTC/USDT | 2h | RSI Mean Reversion | structure_trailing | realistic | 7.588095 |
| BTC/USDT | 2h | RSI Mean Reversion | time_based | high | -42.533713 |
| BTC/USDT | 2h | RSI Mean Reversion | time_based | low | 232.726550 |
| BTC/USDT | 2h | RSI Mean Reversion | time_based | realistic | 232.726550 |
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
| BTC/USDT | 30m | ATR Distance Revert | fixed_atr | high | -120.943606 |
| BTC/USDT | 30m | ATR Distance Revert | fixed_atr | low | -94.294236 |
| BTC/USDT | 30m | ATR Distance Revert | fixed_atr | realistic | -94.294236 |
| BTC/USDT | 30m | ATR Distance Revert | structure_trailing | high | -71.703393 |
| BTC/USDT | 30m | ATR Distance Revert | structure_trailing | low | -48.015831 |
| BTC/USDT | 30m | ATR Distance Revert | structure_trailing | realistic | -48.015831 |
| BTC/USDT | 30m | ATR Distance Revert | time_based | high | -114.820183 |
| BTC/USDT | 30m | ATR Distance Revert | time_based | low | -194.663065 |
| BTC/USDT | 30m | ATR Distance Revert | time_based | realistic | -194.663065 |
| BTC/USDT | 30m | Bollinger Mean Reversion | fixed_atr | high | -119.175635 |
| BTC/USDT | 30m | Bollinger Mean Reversion | fixed_atr | low | -90.823702 |
| BTC/USDT | 30m | Bollinger Mean Reversion | fixed_atr | realistic | -90.823702 |
| BTC/USDT | 30m | Bollinger Mean Reversion | structure_trailing | high | -104.709111 |
| BTC/USDT | 30m | Bollinger Mean Reversion | structure_trailing | low | -81.707675 |
| BTC/USDT | 30m | Bollinger Mean Reversion | structure_trailing | realistic | -81.707675 |
| BTC/USDT | 30m | Bollinger Mean Reversion | time_based | high | -228.579963 |
| BTC/USDT | 30m | Bollinger Mean Reversion | time_based | low | -264.633339 |
| BTC/USDT | 30m | Bollinger Mean Reversion | time_based | realistic | -264.633339 |
| BTC/USDT | 30m | Bollinger SnapBack | fixed_atr | high | -150.983190 |
| BTC/USDT | 30m | Bollinger SnapBack | fixed_atr | low | -118.905922 |
| BTC/USDT | 30m | Bollinger SnapBack | fixed_atr | realistic | -118.905922 |
| BTC/USDT | 30m | Bollinger SnapBack | structure_trailing | high | -117.262574 |
| BTC/USDT | 30m | Bollinger SnapBack | structure_trailing | low | -76.599165 |
| BTC/USDT | 30m | Bollinger SnapBack | structure_trailing | realistic | -76.599165 |
| BTC/USDT | 30m | Bollinger SnapBack | time_based | high | -203.182154 |
| BTC/USDT | 30m | Bollinger SnapBack | time_based | low | -251.819359 |
| BTC/USDT | 30m | Bollinger SnapBack | time_based | realistic | -251.819359 |
| BTC/USDT | 30m | Breakout Retest | fixed_atr | high | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | fixed_atr | low | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | fixed_atr | realistic | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | structure_trailing | high | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | structure_trailing | low | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | structure_trailing | realistic | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | time_based | high | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | time_based | low | 0.000000 |
| BTC/USDT | 30m | Breakout Retest | time_based | realistic | 0.000000 |
| BTC/USDT | 30m | Donchian Breakout | fixed_atr | high | -51.477120 |
| BTC/USDT | 30m | Donchian Breakout | fixed_atr | low | -45.201983 |
| BTC/USDT | 30m | Donchian Breakout | fixed_atr | realistic | -45.201983 |
| BTC/USDT | 30m | Donchian Breakout | structure_trailing | high | -56.193152 |
| BTC/USDT | 30m | Donchian Breakout | structure_trailing | low | -97.093427 |
| BTC/USDT | 30m | Donchian Breakout | structure_trailing | realistic | -97.093427 |
| BTC/USDT | 30m | Donchian Breakout | time_based | high | -7.185445 |
| BTC/USDT | 30m | Donchian Breakout | time_based | low | 25.662562 |
| BTC/USDT | 30m | Donchian Breakout | time_based | realistic | 25.662562 |
| BTC/USDT | 30m | MA SlopePullback | fixed_atr | high | -272.989820 |
| BTC/USDT | 30m | MA SlopePullback | fixed_atr | low | -218.348127 |
| BTC/USDT | 30m | MA SlopePullback | fixed_atr | realistic | -218.348127 |
| BTC/USDT | 30m | MA SlopePullback | structure_trailing | high | -117.022911 |
| BTC/USDT | 30m | MA SlopePullback | structure_trailing | low | -109.946387 |
| BTC/USDT | 30m | MA SlopePullback | structure_trailing | realistic | -109.946387 |
| BTC/USDT | 30m | MA SlopePullback | time_based | high | -470.746298 |
| BTC/USDT | 30m | MA SlopePullback | time_based | low | -396.672847 |
| BTC/USDT | 30m | MA SlopePullback | time_based | realistic | -396.672847 |
| BTC/USDT | 30m | RSI Mean Reversion | fixed_atr | high | -194.163551 |
| BTC/USDT | 30m | RSI Mean Reversion | fixed_atr | low | -161.920431 |
| BTC/USDT | 30m | RSI Mean Reversion | fixed_atr | realistic | -161.920431 |
| BTC/USDT | 30m | RSI Mean Reversion | structure_trailing | high | -101.224793 |
| BTC/USDT | 30m | RSI Mean Reversion | structure_trailing | low | -87.037575 |
| BTC/USDT | 30m | RSI Mean Reversion | structure_trailing | realistic | -87.037575 |
| BTC/USDT | 30m | RSI Mean Reversion | time_based | high | -452.201635 |
| BTC/USDT | 30m | RSI Mean Reversion | time_based | low | -431.960183 |
| BTC/USDT | 30m | RSI Mean Reversion | time_based | realistic | -431.960183 |
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
| ETH/USDT | 15m | ATR Distance Revert | fixed_atr | high | -56.971747 |
| ETH/USDT | 15m | ATR Distance Revert | fixed_atr | low | -62.137671 |
| ETH/USDT | 15m | ATR Distance Revert | fixed_atr | realistic | -62.137671 |
| ETH/USDT | 15m | ATR Distance Revert | structure_trailing | high | -48.569847 |
| ETH/USDT | 15m | ATR Distance Revert | structure_trailing | low | -57.316620 |
| ETH/USDT | 15m | ATR Distance Revert | structure_trailing | realistic | -57.316620 |
| ETH/USDT | 15m | ATR Distance Revert | time_based | high | -112.829099 |
| ETH/USDT | 15m | ATR Distance Revert | time_based | low | -91.030021 |
| ETH/USDT | 15m | ATR Distance Revert | time_based | realistic | -91.030021 |
| ETH/USDT | 15m | Bollinger Mean Reversion | fixed_atr | high | -46.522568 |
| ETH/USDT | 15m | Bollinger Mean Reversion | fixed_atr | low | -50.150940 |
| ETH/USDT | 15m | Bollinger Mean Reversion | fixed_atr | realistic | -50.150940 |
| ETH/USDT | 15m | Bollinger Mean Reversion | structure_trailing | high | -37.469810 |
| ETH/USDT | 15m | Bollinger Mean Reversion | structure_trailing | low | -62.661644 |
| ETH/USDT | 15m | Bollinger Mean Reversion | structure_trailing | realistic | -62.661644 |
| ETH/USDT | 15m | Bollinger Mean Reversion | time_based | high | -108.634978 |
| ETH/USDT | 15m | Bollinger Mean Reversion | time_based | low | -101.472356 |
| ETH/USDT | 15m | Bollinger Mean Reversion | time_based | realistic | -101.472356 |
| ETH/USDT | 15m | Bollinger SnapBack | fixed_atr | high | -51.820999 |
| ETH/USDT | 15m | Bollinger SnapBack | fixed_atr | low | -44.480964 |
| ETH/USDT | 15m | Bollinger SnapBack | fixed_atr | realistic | -44.480964 |
| ETH/USDT | 15m | Bollinger SnapBack | structure_trailing | high | -27.811051 |
| ETH/USDT | 15m | Bollinger SnapBack | structure_trailing | low | -55.136612 |
| ETH/USDT | 15m | Bollinger SnapBack | structure_trailing | realistic | -55.136612 |
| ETH/USDT | 15m | Bollinger SnapBack | time_based | high | -68.165094 |
| ETH/USDT | 15m | Bollinger SnapBack | time_based | low | -72.517094 |
| ETH/USDT | 15m | Bollinger SnapBack | time_based | realistic | -72.517094 |
| ETH/USDT | 15m | Breakout Retest | fixed_atr | high | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | fixed_atr | low | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | structure_trailing | high | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | structure_trailing | low | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | time_based | high | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | time_based | low | 0.000000 |
| ETH/USDT | 15m | Breakout Retest | time_based | realistic | 0.000000 |
| ETH/USDT | 15m | Donchian Breakout | fixed_atr | high | -60.428083 |
| ETH/USDT | 15m | Donchian Breakout | fixed_atr | low | -74.840619 |
| ETH/USDT | 15m | Donchian Breakout | fixed_atr | realistic | -74.840619 |
| ETH/USDT | 15m | Donchian Breakout | structure_trailing | high | -26.550496 |
| ETH/USDT | 15m | Donchian Breakout | structure_trailing | low | -56.365171 |
| ETH/USDT | 15m | Donchian Breakout | structure_trailing | realistic | -56.365171 |
| ETH/USDT | 15m | Donchian Breakout | time_based | high | -158.616958 |
| ETH/USDT | 15m | Donchian Breakout | time_based | low | -197.569961 |
| ETH/USDT | 15m | Donchian Breakout | time_based | realistic | -197.569961 |
| ETH/USDT | 15m | MA SlopePullback | fixed_atr | high | -63.706640 |
| ETH/USDT | 15m | MA SlopePullback | fixed_atr | low | -54.645252 |
| ETH/USDT | 15m | MA SlopePullback | fixed_atr | realistic | -54.645252 |
| ETH/USDT | 15m | MA SlopePullback | structure_trailing | high | -14.812296 |
| ETH/USDT | 15m | MA SlopePullback | structure_trailing | low | -49.767203 |
| ETH/USDT | 15m | MA SlopePullback | structure_trailing | realistic | -49.767203 |
| ETH/USDT | 15m | MA SlopePullback | time_based | high | -76.393327 |
| ETH/USDT | 15m | MA SlopePullback | time_based | low | -99.707843 |
| ETH/USDT | 15m | MA SlopePullback | time_based | realistic | -99.707843 |
| ETH/USDT | 15m | RSI Mean Reversion | fixed_atr | high | -71.215558 |
| ETH/USDT | 15m | RSI Mean Reversion | fixed_atr | low | -79.981947 |
| ETH/USDT | 15m | RSI Mean Reversion | fixed_atr | realistic | -79.981947 |
| ETH/USDT | 15m | RSI Mean Reversion | structure_trailing | high | -49.192507 |
| ETH/USDT | 15m | RSI Mean Reversion | structure_trailing | low | -58.405864 |
| ETH/USDT | 15m | RSI Mean Reversion | structure_trailing | realistic | -58.405864 |
| ETH/USDT | 15m | RSI Mean Reversion | time_based | high | -78.952942 |
| ETH/USDT | 15m | RSI Mean Reversion | time_based | low | -67.853629 |
| ETH/USDT | 15m | RSI Mean Reversion | time_based | realistic | -67.853629 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | fixed_atr | high | -101.143746 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | fixed_atr | low | -65.640810 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | fixed_atr | realistic | -65.640810 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | structure_trailing | high | -46.057130 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | structure_trailing | low | -72.503771 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | structure_trailing | realistic | -72.503771 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | time_based | high | -453.841335 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | time_based | low | -447.569726 |
| ETH/USDT | 15m | Range Breakout w/ EMA Trend Filter | time_based | realistic | -447.569726 |
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
| ETH/USDT | 1h | ATR Distance Revert | fixed_atr | high | -353.907004 |
| ETH/USDT | 1h | ATR Distance Revert | fixed_atr | low | -251.713526 |
| ETH/USDT | 1h | ATR Distance Revert | fixed_atr | realistic | -251.713526 |
| ETH/USDT | 1h | ATR Distance Revert | structure_trailing | high | -265.542227 |
| ETH/USDT | 1h | ATR Distance Revert | structure_trailing | low | -188.145081 |
| ETH/USDT | 1h | ATR Distance Revert | structure_trailing | realistic | -188.145081 |
| ETH/USDT | 1h | ATR Distance Revert | time_based | high | 23.563990 |
| ETH/USDT | 1h | ATR Distance Revert | time_based | low | -97.701468 |
| ETH/USDT | 1h | ATR Distance Revert | time_based | realistic | -97.701468 |
| ETH/USDT | 1h | Bollinger Mean Reversion | fixed_atr | high | -219.884533 |
| ETH/USDT | 1h | Bollinger Mean Reversion | fixed_atr | low | -190.224277 |
| ETH/USDT | 1h | Bollinger Mean Reversion | fixed_atr | realistic | -190.224277 |
| ETH/USDT | 1h | Bollinger Mean Reversion | structure_trailing | high | -186.494815 |
| ETH/USDT | 1h | Bollinger Mean Reversion | structure_trailing | low | -126.047285 |
| ETH/USDT | 1h | Bollinger Mean Reversion | structure_trailing | realistic | -126.047285 |
| ETH/USDT | 1h | Bollinger Mean Reversion | time_based | high | -441.786882 |
| ETH/USDT | 1h | Bollinger Mean Reversion | time_based | low | -416.245309 |
| ETH/USDT | 1h | Bollinger Mean Reversion | time_based | realistic | -416.245309 |
| ETH/USDT | 1h | Bollinger SnapBack | fixed_atr | high | -219.884533 |
| ETH/USDT | 1h | Bollinger SnapBack | fixed_atr | low | -190.224277 |
| ETH/USDT | 1h | Bollinger SnapBack | fixed_atr | realistic | -190.224277 |
| ETH/USDT | 1h | Bollinger SnapBack | structure_trailing | high | -186.494815 |
| ETH/USDT | 1h | Bollinger SnapBack | structure_trailing | low | -126.047285 |
| ETH/USDT | 1h | Bollinger SnapBack | structure_trailing | realistic | -126.047285 |
| ETH/USDT | 1h | Bollinger SnapBack | time_based | high | -441.786882 |
| ETH/USDT | 1h | Bollinger SnapBack | time_based | low | -416.245309 |
| ETH/USDT | 1h | Bollinger SnapBack | time_based | realistic | -416.245309 |
| ETH/USDT | 1h | Breakout Retest | fixed_atr | high | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | fixed_atr | low | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | structure_trailing | high | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | structure_trailing | low | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | time_based | high | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | time_based | low | 0.000000 |
| ETH/USDT | 1h | Breakout Retest | time_based | realistic | 0.000000 |
| ETH/USDT | 1h | Donchian Breakout | fixed_atr | high | -207.716293 |
| ETH/USDT | 1h | Donchian Breakout | fixed_atr | low | -134.659536 |
| ETH/USDT | 1h | Donchian Breakout | fixed_atr | realistic | -134.659536 |
| ETH/USDT | 1h | Donchian Breakout | structure_trailing | high | -107.186609 |
| ETH/USDT | 1h | Donchian Breakout | structure_trailing | low | -145.199110 |
| ETH/USDT | 1h | Donchian Breakout | structure_trailing | realistic | -145.199110 |
| ETH/USDT | 1h | Donchian Breakout | time_based | high | -252.102491 |
| ETH/USDT | 1h | Donchian Breakout | time_based | low | -247.441527 |
| ETH/USDT | 1h | Donchian Breakout | time_based | realistic | -247.441527 |
| ETH/USDT | 1h | MA SlopePullback | fixed_atr | high | -259.691800 |
| ETH/USDT | 1h | MA SlopePullback | fixed_atr | low | -282.993048 |
| ETH/USDT | 1h | MA SlopePullback | fixed_atr | realistic | -282.993048 |
| ETH/USDT | 1h | MA SlopePullback | structure_trailing | high | 28.170770 |
| ETH/USDT | 1h | MA SlopePullback | structure_trailing | low | 33.157672 |
| ETH/USDT | 1h | MA SlopePullback | structure_trailing | realistic | 33.157672 |
| ETH/USDT | 1h | MA SlopePullback | time_based | high | -8.076900 |
| ETH/USDT | 1h | MA SlopePullback | time_based | low | -8.076900 |
| ETH/USDT | 1h | MA SlopePullback | time_based | realistic | -8.076900 |
| ETH/USDT | 1h | RSI Mean Reversion | fixed_atr | high | -208.990174 |
| ETH/USDT | 1h | RSI Mean Reversion | fixed_atr | low | -189.785077 |
| ETH/USDT | 1h | RSI Mean Reversion | fixed_atr | realistic | -189.785077 |
| ETH/USDT | 1h | RSI Mean Reversion | structure_trailing | high | -157.103943 |
| ETH/USDT | 1h | RSI Mean Reversion | structure_trailing | low | -140.721111 |
| ETH/USDT | 1h | RSI Mean Reversion | structure_trailing | realistic | -140.721111 |
| ETH/USDT | 1h | RSI Mean Reversion | time_based | high | -247.077721 |
| ETH/USDT | 1h | RSI Mean Reversion | time_based | low | -316.701125 |
| ETH/USDT | 1h | RSI Mean Reversion | time_based | realistic | -316.701125 |
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
| ETH/USDT | 2h | Bollinger Mean Reversion | fixed_atr | high | -444.220330 |
| ETH/USDT | 2h | Bollinger Mean Reversion | fixed_atr | low | -330.023761 |
| ETH/USDT | 2h | Bollinger Mean Reversion | fixed_atr | realistic | -330.023761 |
| ETH/USDT | 2h | Bollinger Mean Reversion | structure_trailing | high | -273.117844 |
| ETH/USDT | 2h | Bollinger Mean Reversion | structure_trailing | low | -170.549292 |
| ETH/USDT | 2h | Bollinger Mean Reversion | structure_trailing | realistic | -170.549292 |
| ETH/USDT | 2h | Bollinger Mean Reversion | time_based | high | -156.890913 |
| ETH/USDT | 2h | Bollinger Mean Reversion | time_based | low | -156.890913 |
| ETH/USDT | 2h | Bollinger Mean Reversion | time_based | realistic | -156.890913 |
| ETH/USDT | 2h | Bollinger SnapBack | fixed_atr | high | -444.220330 |
| ETH/USDT | 2h | Bollinger SnapBack | fixed_atr | low | -330.023761 |
| ETH/USDT | 2h | Bollinger SnapBack | fixed_atr | realistic | -330.023761 |
| ETH/USDT | 2h | Bollinger SnapBack | structure_trailing | high | -273.117844 |
| ETH/USDT | 2h | Bollinger SnapBack | structure_trailing | low | -170.549292 |
| ETH/USDT | 2h | Bollinger SnapBack | structure_trailing | realistic | -170.549292 |
| ETH/USDT | 2h | Bollinger SnapBack | time_based | high | -156.890913 |
| ETH/USDT | 2h | Bollinger SnapBack | time_based | low | -156.890913 |
| ETH/USDT | 2h | Bollinger SnapBack | time_based | realistic | -156.890913 |
| ETH/USDT | 2h | Breakout Retest | fixed_atr | high | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | fixed_atr | low | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | structure_trailing | high | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | structure_trailing | low | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | time_based | high | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | time_based | low | 0.000000 |
| ETH/USDT | 2h | Breakout Retest | time_based | realistic | 0.000000 |
| ETH/USDT | 2h | Donchian Breakout | fixed_atr | high | -181.716734 |
| ETH/USDT | 2h | Donchian Breakout | fixed_atr | low | -248.845119 |
| ETH/USDT | 2h | Donchian Breakout | fixed_atr | realistic | -248.845119 |
| ETH/USDT | 2h | Donchian Breakout | structure_trailing | high | -72.627523 |
| ETH/USDT | 2h | Donchian Breakout | structure_trailing | low | -168.600093 |
| ETH/USDT | 2h | Donchian Breakout | structure_trailing | realistic | -168.600093 |
| ETH/USDT | 2h | Donchian Breakout | time_based | high | 96.258132 |
| ETH/USDT | 2h | Donchian Breakout | time_based | low | 96.258132 |
| ETH/USDT | 2h | Donchian Breakout | time_based | realistic | 96.258132 |
| ETH/USDT | 2h | MA SlopePullback | fixed_atr | high | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | fixed_atr | low | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | structure_trailing | high | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | structure_trailing | low | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | time_based | high | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | time_based | low | 0.000000 |
| ETH/USDT | 2h | MA SlopePullback | time_based | realistic | 0.000000 |
| ETH/USDT | 2h | RSI Mean Reversion | fixed_atr | high | -268.941790 |
| ETH/USDT | 2h | RSI Mean Reversion | fixed_atr | low | -198.888511 |
| ETH/USDT | 2h | RSI Mean Reversion | fixed_atr | realistic | -198.888511 |
| ETH/USDT | 2h | RSI Mean Reversion | structure_trailing | high | -236.809590 |
| ETH/USDT | 2h | RSI Mean Reversion | structure_trailing | low | -45.717804 |
| ETH/USDT | 2h | RSI Mean Reversion | structure_trailing | realistic | -45.717804 |
| ETH/USDT | 2h | RSI Mean Reversion | time_based | high | 16.576454 |
| ETH/USDT | 2h | RSI Mean Reversion | time_based | low | -80.804840 |
| ETH/USDT | 2h | RSI Mean Reversion | time_based | realistic | -80.804840 |
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
| ETH/USDT | 30m | ATR Distance Revert | fixed_atr | high | -178.046297 |
| ETH/USDT | 30m | ATR Distance Revert | fixed_atr | low | -167.300982 |
| ETH/USDT | 30m | ATR Distance Revert | fixed_atr | realistic | -167.300982 |
| ETH/USDT | 30m | ATR Distance Revert | structure_trailing | high | -36.607823 |
| ETH/USDT | 30m | ATR Distance Revert | structure_trailing | low | -87.654760 |
| ETH/USDT | 30m | ATR Distance Revert | structure_trailing | realistic | -87.654760 |
| ETH/USDT | 30m | ATR Distance Revert | time_based | high | -216.239000 |
| ETH/USDT | 30m | ATR Distance Revert | time_based | low | -152.171043 |
| ETH/USDT | 30m | ATR Distance Revert | time_based | realistic | -152.171043 |
| ETH/USDT | 30m | Bollinger Mean Reversion | fixed_atr | high | 50.923043 |
| ETH/USDT | 30m | Bollinger Mean Reversion | fixed_atr | low | -63.846698 |
| ETH/USDT | 30m | Bollinger Mean Reversion | fixed_atr | realistic | -63.846698 |
| ETH/USDT | 30m | Bollinger Mean Reversion | structure_trailing | high | -30.046351 |
| ETH/USDT | 30m | Bollinger Mean Reversion | structure_trailing | low | -82.024415 |
| ETH/USDT | 30m | Bollinger Mean Reversion | structure_trailing | realistic | -82.024415 |
| ETH/USDT | 30m | Bollinger Mean Reversion | time_based | high | -49.851333 |
| ETH/USDT | 30m | Bollinger Mean Reversion | time_based | low | -88.080642 |
| ETH/USDT | 30m | Bollinger Mean Reversion | time_based | realistic | -88.080642 |
| ETH/USDT | 30m | Bollinger SnapBack | fixed_atr | high | 49.501735 |
| ETH/USDT | 30m | Bollinger SnapBack | fixed_atr | low | -13.907317 |
| ETH/USDT | 30m | Bollinger SnapBack | fixed_atr | realistic | -13.907317 |
| ETH/USDT | 30m | Bollinger SnapBack | structure_trailing | high | -39.964231 |
| ETH/USDT | 30m | Bollinger SnapBack | structure_trailing | low | -75.823094 |
| ETH/USDT | 30m | Bollinger SnapBack | structure_trailing | realistic | -75.823094 |
| ETH/USDT | 30m | Bollinger SnapBack | time_based | high | -131.833914 |
| ETH/USDT | 30m | Bollinger SnapBack | time_based | low | -103.625592 |
| ETH/USDT | 30m | Bollinger SnapBack | time_based | realistic | -103.625592 |
| ETH/USDT | 30m | Breakout Retest | fixed_atr | high | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | fixed_atr | low | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | fixed_atr | realistic | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | structure_trailing | high | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | structure_trailing | low | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | structure_trailing | realistic | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | time_based | high | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | time_based | low | 0.000000 |
| ETH/USDT | 30m | Breakout Retest | time_based | realistic | 0.000000 |
| ETH/USDT | 30m | Donchian Breakout | fixed_atr | high | -158.321154 |
| ETH/USDT | 30m | Donchian Breakout | fixed_atr | low | -135.787016 |
| ETH/USDT | 30m | Donchian Breakout | fixed_atr | realistic | -135.787016 |
| ETH/USDT | 30m | Donchian Breakout | structure_trailing | high | -40.305922 |
| ETH/USDT | 30m | Donchian Breakout | structure_trailing | low | -82.429688 |
| ETH/USDT | 30m | Donchian Breakout | structure_trailing | realistic | -82.429688 |
| ETH/USDT | 30m | Donchian Breakout | time_based | high | -262.477866 |
| ETH/USDT | 30m | Donchian Breakout | time_based | low | -232.592043 |
| ETH/USDT | 30m | Donchian Breakout | time_based | realistic | -232.592043 |
| ETH/USDT | 30m | MA SlopePullback | fixed_atr | high | 40.274684 |
| ETH/USDT | 30m | MA SlopePullback | fixed_atr | low | 54.262318 |
| ETH/USDT | 30m | MA SlopePullback | fixed_atr | realistic | 54.262318 |
| ETH/USDT | 30m | MA SlopePullback | structure_trailing | high | 121.903200 |
| ETH/USDT | 30m | MA SlopePullback | structure_trailing | low | -21.593602 |
| ETH/USDT | 30m | MA SlopePullback | structure_trailing | realistic | -21.593602 |
| ETH/USDT | 30m | MA SlopePullback | time_based | high | 11.044237 |
| ETH/USDT | 30m | MA SlopePullback | time_based | low | 21.266809 |
| ETH/USDT | 30m | MA SlopePullback | time_based | realistic | 21.266809 |
| ETH/USDT | 30m | RSI Mean Reversion | fixed_atr | high | -113.597620 |
| ETH/USDT | 30m | RSI Mean Reversion | fixed_atr | low | -104.858264 |
| ETH/USDT | 30m | RSI Mean Reversion | fixed_atr | realistic | -104.858264 |
| ETH/USDT | 30m | RSI Mean Reversion | structure_trailing | high | -3.210356 |
| ETH/USDT | 30m | RSI Mean Reversion | structure_trailing | low | -24.122785 |
| ETH/USDT | 30m | RSI Mean Reversion | structure_trailing | realistic | -24.122785 |
| ETH/USDT | 30m | RSI Mean Reversion | time_based | high | -206.120323 |
| ETH/USDT | 30m | RSI Mean Reversion | time_based | low | -232.779138 |
| ETH/USDT | 30m | RSI Mean Reversion | time_based | realistic | -232.779138 |
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
| ETH/USDT | 4h | RSI Mean Reversion | fixed_atr | high | 514.181637 |
| ETH/USDT | 4h | RSI Mean Reversion | fixed_atr | low | 435.602869 |
| ETH/USDT | 4h | RSI Mean Reversion | fixed_atr | realistic | 435.602869 |
| ETH/USDT | 4h | RSI Mean Reversion | structure_trailing | high | 436.838380 |
| ETH/USDT | 4h | RSI Mean Reversion | structure_trailing | low | 33.896124 |
| ETH/USDT | 4h | RSI Mean Reversion | structure_trailing | realistic | 33.896124 |
| ETH/USDT | 4h | RSI Mean Reversion | time_based | high | 514.181637 |
| ETH/USDT | 4h | RSI Mean Reversion | time_based | low | 435.602869 |
| ETH/USDT | 4h | RSI Mean Reversion | time_based | realistic | 435.602869 |
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
