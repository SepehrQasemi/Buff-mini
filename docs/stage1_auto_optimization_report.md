# Stage-1 Auto Optimization Report

- run_id: `20260226_120451_fa8f0e044192_stage1`
- stage: `stage1`
- runtime_seconds: `1288.38`
- seed: `42`
- config_hash: `fa8f0e044192`
- data_hash: `c923139f4ab67059`
- cost(round_trip_cost_pct): `0.1`
- candidates A/B/C: `1743/100/20`

## Top 3 Candidates
### Rank 1 - TrendPullback
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.238113302469894, "atr_tp_multiplier": 4.808126794212899, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 100, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 12, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 20, "rsi_short_entry": 74, "trailing_atr_k": 1.2290405402496654}`
- Holdout metrics: trade_count=2, PF=10.0000, expectancy=298.2824, max_dd=0.0005, return_pct=0.0298
- Holdout range: `2025-07-10T22:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 2 - TrendPullback
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.9818651992020315, "atr_tp_multiplier": 4.640358696520796, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 55, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 96, "regime_gate_long": false, "regime_gate_short": false, "rsi_long_entry": 22, "rsi_short_entry": 76, "trailing_atr_k": 1.2715147760052585}`
- Holdout metrics: trade_count=1, PF=10.0000, expectancy=132.9252, max_dd=0.0003, return_pct=0.0133
- Holdout range: `2025-07-10T22:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 3 - TrendPullback
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.2325296404154624, "atr_tp_multiplier": 3.5692219187737475, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 55, "ema_fast": 50, "ema_slow": 100, "max_holding_bars": 24, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 20, "rsi_short_entry": 76, "trailing_atr_k": 1.6478445197435425}`
- Holdout metrics: trade_count=1, PF=10.0000, expectancy=100.5105, max_dd=0.0003, return_pct=0.0101
- Holdout range: `2025-07-10T22:00:00+00:00..2026-02-26T09:00:00+00:00`
