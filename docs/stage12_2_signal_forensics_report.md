# Stage-12.2 Signal Forensics & Context Modeling Foundation

- run_id: `20260301_060514_8bb2fcf2c8c0_stage12`
- total_trades: `720`
- winning_trades: `308`
- losing_trades: `412`
- avg_context_score_winners: `0.248546`
- avg_context_score_losers: `0.242876`
- context_score_diff: `0.005670`
- context_separation_effect_size: `0.040543`
- context_separation_detected: `False`
- final_stage12_2_verdict: `RANDOM_NOISE`

## Winner vs Loser Separation by Strategy
| strategy | samples_win | samples_loss | mean_context_win | mean_context_loss | separation_score |
| --- | ---: | ---: | ---: | ---: | ---: |
| ATR Distance Revert | 42 | 42 | 0.248907 | 0.208054 | 0.580467 |
| Bollinger Mean Reversion | 64 | 59 | 0.226037 | 0.233448 | -0.087991 |
| Bollinger SnapBack | 73 | 56 | 0.180136 | 0.237872 | -0.615734 |
| Donchian Breakout | 22 | 95 | 0.484792 | 0.189624 | 1.694777 |
| MA SlopePullback | 36 | 81 | 0.212152 | 0.283867 | -0.476864 |
| RSI Mean Reversion | 44 | 52 | 0.263033 | 0.234542 | 0.360288 |
| Range Breakout w/ EMA Trend Filter | 15 | 12 | 0.431531 | 0.420439 | 0.050992 |
| Range Fade | 9 | 0 | 0.166222 | 0.000000 | 0.000000 |
| Trend Pullback | 3 | 15 | 0.212107 | 0.398879 | -1.246933 |
