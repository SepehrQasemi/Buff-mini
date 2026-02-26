# Stage-1 Auto Optimization Report

- run_id: `20260226_170035_3fc0a2b58f96_stage1`
- stage: `stage1`
- runtime_seconds: `1763.95`
- seed: `42`
- config_hash: `3fc0a2b58f96`
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
- Parameters: `{"atr_sl_multiplier": 1.5800351169804272, "atr_tp_multiplier": 4.405036104697668, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 100, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 96, "regime_gate_long": false, "regime_gate_short": false, "rsi_long_entry": 40, "rsi_short_entry": 80, "trailing_atr_k": 1.4568945269920612}`
- Holdout metrics: trade_count=12, tpm=1.5621, pf_adj=1.5885, PF=4.0404, expectancy=133.1680, exp_lcb=59.9725, effective_edge=93.6833, exposure_ratio=0.0241, low_signal_penalty=0.4024, penalty_relief=True, max_dd=0.0631, return_pct=0.0799
- Holdout range: `2025-07-10T22:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 2 - TrendPullback
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.238113302469894, "atr_tp_multiplier": 4.808126794212899, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 100, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 12, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 20, "rsi_short_entry": 74, "trailing_atr_k": 1.2290405402496654}`
- Holdout metrics: trade_count=2, tpm=0.2604, pf_adj=1.3462, PF=10.0000, expectancy=298.2824, exp_lcb=282.5295, effective_edge=73.5568, exposure_ratio=0.0014, low_signal_penalty=0.4837, penalty_relief=True, max_dd=0.0005, return_pct=0.0298
- Holdout range: `2025-07-10T22:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 3 - TrendPullback
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.2114282453293925, "atr_tp_multiplier": 1.445540427428266, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 55, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 24, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 25, "rsi_short_entry": 66, "trailing_atr_k": 1.8664674198625655}`
- Holdout metrics: trade_count=6, tpm=0.7811, pf_adj=1.9643, PF=10.0000, expectancy=106.5357, exp_lcb=87.1584, effective_edge=68.0753, exposure_ratio=0.0025, low_signal_penalty=0.4512, penalty_relief=True, max_dd=0.0005, return_pct=0.0320
- Holdout range: `2025-07-10T22:00:00+00:00..2026-02-26T09:00:00+00:00`
