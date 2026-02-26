# Stage-1 Auto Optimization Report

- run_id: `20260226_110627_275a1c8009b5_stage1`
- stage: `stage1`
- runtime_seconds: `151.92`
- seed: `42`
- config_hash: `275a1c8009b5`
- data_hash: `7c0aee7f5c04109b`
- cost(round_trip_cost_pct): `0.1`
- candidates A/B/C: `600/100/20`

## Top 3 Candidates
### Rank 1 - RangeBreakoutTrendFilter
- Strategy: `Range Breakout w/ EMA Trend Filter`
- Gating: `none`
- Exit mode: `breakeven_1r`
- Entry rules: Long when close > Donchian(period) high and EMA_fast > EMA_slow; short when close < Donchian(period) low and EMA_fast < EMA_slow.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.2526325518213914, "atr_tp_multiplier": 1.0831472979033663, "bollinger_period": 20, "bollinger_std": 2.0, "channel_period": 100, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 24, "regime_gate_long": true, "regime_gate_short": true, "rsi_long_entry": 30, "rsi_short_entry": 71, "trailing_atr_k": 1.2936574033145871}`
- Holdout metrics: trade_count=100, PF=1.1566, expectancy=11.1239, max_dd=0.0829, return_pct=0.0710
- Holdout range: `2023-01-01T00:00:00+00:00..2023-04-10T23:00:00+00:00`

### Rank 2 - TrendPullback
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.3803182524121915, "atr_tp_multiplier": 1.4607442132570427, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 55, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 24, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 37, "rsi_short_entry": 66, "trailing_atr_k": 1.845793605206971}`
- Holdout metrics: trade_count=8, PF=2.6532, expectancy=32.7067, max_dd=0.0327, return_pct=0.0131
- Holdout range: `2023-01-01T00:00:00+00:00..2023-04-10T23:00:00+00:00`

### Rank 3 - DonchianBreakout
- Strategy: `Donchian Breakout`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Entry rules: Long when close > Donchian(period) high; short when close < Donchian(period) low.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.4252793211884693, "atr_tp_multiplier": 1.6176923520885698, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 55, "ema_fast": 50, "ema_slow": 100, "max_holding_bars": 24, "regime_gate_long": false, "regime_gate_short": false, "rsi_long_entry": 35, "rsi_short_entry": 65, "trailing_atr_k": 1.2965004733819594}`
- Holdout metrics: trade_count=83, PF=1.7987, expectancy=60.5040, max_dd=0.0662, return_pct=0.3053
- Holdout range: `2023-01-01T00:00:00+00:00..2023-04-10T23:00:00+00:00`
