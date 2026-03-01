# Stage-12.2 Signal Forensics & Context Modeling Foundation

- run_id: `20260301_041932_2fc2823beb24_stage12`
- total_trades: `3328`
- winning_trades: `1261`
- losing_trades: `2067`
- avg_context_score_winners: `0.202218`
- avg_context_score_losers: `0.200770`
- context_score_diff: `0.001448`
- context_separation_effect_size: `0.011013`
- context_separation_detected: `False`
- final_stage12_2_verdict: `RANDOM_NOISE`

## Winner vs Loser Separation by Strategy
| strategy | samples_win | samples_loss | mean_context_win | mean_context_loss | separation_score |
| --- | ---: | ---: | ---: | ---: | ---: |
| ATR Distance Revert | 199 | 343 | 0.240056 | 0.217507 | 0.222693 |
| Bollinger Mean Reversion | 235 | 383 | 0.211630 | 0.193379 | 0.159958 |
| Bollinger SnapBack | 226 | 322 | 0.205169 | 0.193618 | 0.099553 |
| Donchian Breakout | 222 | 336 | 0.164686 | 0.164097 | 0.003311 |
| MA SlopePullback | 86 | 106 | 0.185548 | 0.288657 | -0.789182 |
| RSI Mean Reversion | 260 | 529 | 0.190985 | 0.194852 | -0.034530 |
| Range Breakout w/ EMA Trend Filter | 21 | 27 | 0.324703 | 0.322064 | 0.012650 |
| Range Fade | 9 | 0 | 0.166222 | 0.000000 | 0.000000 |
| Trend Pullback | 3 | 21 | 0.212107 | 0.308167 | -0.506522 |
