# Stage-1 Auto Optimization Report

- run_id: `20260226_141335_3fc0a2b58f96_stage1`
- stage: `stage1`
- runtime_seconds: `1847.50`
- seed: `42`
- config_hash: `3fc0a2b58f96`
- data_hash: `c923139f4ab67059`
- cost(round_trip_cost_pct): `0.1`
- candidates A/B/C: `1743/100/20`

## Top 3 Candidates
### Rank 1 - TrendPullback
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.2833113852489486, "atr_tp_multiplier": 3.2421489330660194, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 20, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 12, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 24, "rsi_short_entry": 70, "trailing_atr_k": 1.606758848750133}`
- Holdout metrics: trade_count=2, tpm=0.2604, PF=10.0000, expectancy=250.8928, low_signal_penalty=0.4837, penalty_relief=True, max_dd=0.0005, return_pct=0.0251
- Holdout range: `2025-07-10T22:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 2 - TrendPullback
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.238113302469894, "atr_tp_multiplier": 4.808126794212899, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 100, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 12, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 20, "rsi_short_entry": 74, "trailing_atr_k": 1.2290405402496654}`
- Holdout metrics: trade_count=2, tpm=0.2604, PF=10.0000, expectancy=298.2824, low_signal_penalty=0.4837, penalty_relief=True, max_dd=0.0005, return_pct=0.0298
- Holdout range: `2025-07-10T22:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 3 - TrendPullback
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.0749720144899566, "atr_tp_multiplier": 2.385971445252195, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 100, "ema_fast": 50, "ema_slow": 100, "max_holding_bars": 24, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 23, "rsi_short_entry": 66, "trailing_atr_k": 1.1858046622963316}`
- Holdout metrics: trade_count=6, tpm=0.7811, PF=10.0000, expectancy=166.4198, low_signal_penalty=0.4512, penalty_relief=True, max_dd=0.0076, return_pct=0.0384
- Holdout range: `2025-07-10T22:00:00+00:00..2026-02-26T09:00:00+00:00`
