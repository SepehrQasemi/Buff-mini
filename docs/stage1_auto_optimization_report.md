# Stage-1 Auto Optimization Report

- run_id: `20260226_223856_417c7a3f42c9_stage1`
- stage: `stage1`
- runtime_seconds: `2163.51`
- seed: `42`
- config_hash: `417c7a3f42c9`
- data_hash: `c923139f4ab67059`
- cost(round_trip_cost_pct): `0.1`
- candidates A/B/C: `1743/100/50`
- accepted_count: `9`
- near_miss_count: `1`
- stage_c_seconds: `629.36`

## Top 3 Candidates
### Rank 1 - TrendPullback
- Strategy: `Trend Pullback`
- Gating: `none`
- Exit mode: `fixed_atr`
- Holdout months used: `3`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.267671495835672, "atr_tp_multiplier": 4.136969411756131, "bollinger_period": 20, "bollinger_std": 2.0, "channel_period": 100, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 48, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 35, "rsi_short_entry": 64, "trailing_atr_k": 2.0019281371341706}`
- Holdout metrics: trade_count=28, tpm=9.1304, pf_adj=1.4936, PF=2.3750, expectancy=71.8697, exp_lcb=30.5615, effective_edge=279.0398, exposure_ratio=0.1073, low_signal_penalty=0.0000, penalty_relief=False, max_dd=0.1093, return_pct=0.1006
- Holdout range: `2025-11-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 2 - TrendPullback
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Holdout months used: `3`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.2550028496017491, "atr_tp_multiplier": 4.098810372044909, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 55, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 48, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 35, "rsi_short_entry": 64, "trailing_atr_k": 1.4967991102093166}`
- Holdout metrics: trade_count=8, tpm=2.6087, pf_adj=2.2414, PF=10.0000, expectancy=119.5705, exp_lcb=58.7705, effective_edge=153.3145, exposure_ratio=0.0251, low_signal_penalty=0.6739, penalty_relief=False, max_dd=0.0346, return_pct=0.0478
- Holdout range: `2025-11-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 3 - TrendPullback
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Holdout months used: `3`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.990928112487736, "atr_tp_multiplier": 2.615986332726116, "bollinger_period": 20, "bollinger_std": 2.0, "channel_period": 20, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 48, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 31, "rsi_short_entry": 62, "trailing_atr_k": 1.6090443409457977}`
- Holdout metrics: trade_count=8, tpm=2.6087, pf_adj=2.2414, PF=10.0000, expectancy=98.7015, exp_lcb=39.3389, effective_edge=102.6232, exposure_ratio=0.0204, low_signal_penalty=0.6739, penalty_relief=False, max_dd=0.0582, return_pct=0.0395
- Holdout range: `2025-11-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`
