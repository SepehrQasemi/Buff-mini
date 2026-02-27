# Stage-1 Auto Optimization Report

- run_id: `20260226_234139_b78b63b21ba0_stage1`
- stage: `stage1`
- runtime_seconds: `1562.19`
- seed: `42`
- config_hash: `b78b63b21ba0`
- data_hash: `c923139f4ab67059`
- cost(round_trip_cost_pct): `0.1`
- candidates A/B/C: `1743/100/50`
- Tier A count: `1`
- Tier B count: `0`
- near_miss_count: `29`
- stage_c_seconds: `551.50`

## Threshold-Selected Candidates
### Rank 1 - TrendPullback (Tier A)
- Strategy: `Trend Pullback`
- Gating: `none`
- Exit mode: `fixed_atr`
- Holdout months used: `3`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.267671495835672, "atr_tp_multiplier": 4.136969411756131, "bollinger_period": 20, "bollinger_std": 2.0, "channel_period": 100, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 48, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 35, "rsi_short_entry": 64, "trailing_atr_k": 2.0019281371341706}`
- Holdout metrics: trade_count=28, tpm=9.1304, pf_adj=1.4936, PF=2.3750, expectancy=71.8697, exp_lcb=30.5615, effective_edge=279.0398, exposure_ratio=0.1073, low_signal_penalty=0.0000, penalty_relief=False, max_dd=0.1093, return_pct=0.1006
- Holdout range: `2025-11-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 2 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Holdout months used: `3`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.2550028496017491, "atr_tp_multiplier": 4.098810372044909, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 55, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 48, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 35, "rsi_short_entry": 64, "trailing_atr_k": 1.4967991102093166}`
- Holdout metrics: trade_count=8, tpm=2.6087, pf_adj=2.2414, PF=10.0000, expectancy=119.5705, exp_lcb=58.7705, effective_edge=153.3145, exposure_ratio=0.0251, low_signal_penalty=0.6739, penalty_relief=False, max_dd=0.0346, return_pct=0.0478
- Holdout range: `2025-11-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 3 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `breakeven_1r`
- Holdout months used: `3`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.3998430132798356, "atr_tp_multiplier": 4.158633815054241, "bollinger_period": 20, "bollinger_std": 2.0, "channel_period": 100, "ema_fast": 50, "ema_slow": 100, "max_holding_bars": 24, "regime_gate_long": true, "regime_gate_short": true, "rsi_long_entry": 28, "rsi_short_entry": 60, "trailing_atr_k": 1.0414705376315427}`
- Holdout metrics: trade_count=9, tpm=2.9348, pf_adj=2.3729, PF=10.0000, expectancy=87.3513, exp_lcb=44.3964, effective_edge=130.2938, exposure_ratio=0.0326, low_signal_penalty=0.6332, penalty_relief=False, max_dd=0.0551, return_pct=0.0393
- Holdout range: `2025-11-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 4 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Holdout months used: `3`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.990928112487736, "atr_tp_multiplier": 2.615986332726116, "bollinger_period": 20, "bollinger_std": 2.0, "channel_period": 20, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 48, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 31, "rsi_short_entry": 62, "trailing_atr_k": 1.6090443409457977}`
- Holdout metrics: trade_count=8, tpm=2.6087, pf_adj=2.2414, PF=10.0000, expectancy=98.7015, exp_lcb=39.3389, effective_edge=102.6232, exposure_ratio=0.0204, low_signal_penalty=0.6739, penalty_relief=False, max_dd=0.0582, return_pct=0.0395
- Holdout range: `2025-11-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 5 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.7425133156581634, "atr_tp_multiplier": 1.8876025851752942, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 55, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 48, "regime_gate_long": false, "regime_gate_short": false, "rsi_long_entry": 34, "rsi_short_entry": 65, "trailing_atr_k": 2.0206465916386844}`
- Holdout metrics: trade_count=16, tpm=1.3151, pf_adj=1.7578, PF=4.1257, expectancy=86.4935, exp_lcb=43.5536, effective_edge=57.2759, exposure_ratio=0.0100, low_signal_penalty=0.8356, penalty_relief=False, max_dd=0.0261, return_pct=0.0692
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 6 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.0935789133630474, "atr_tp_multiplier": 1.3910929559521508, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 100, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 48, "regime_gate_long": true, "regime_gate_short": true, "rsi_long_entry": 24, "rsi_short_entry": 63, "trailing_atr_k": 1.6871564242745687}`
- Holdout metrics: trade_count=16, tpm=1.3151, pf_adj=1.4816, PF=2.9865, expectancy=88.6320, exp_lcb=42.3147, effective_edge=55.6467, exposure_ratio=0.0104, low_signal_penalty=0.8356, penalty_relief=False, max_dd=0.0309, return_pct=0.0709
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 7 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.238113302469894, "atr_tp_multiplier": 4.808126794212899, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 100, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 12, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 20, "rsi_short_entry": 74, "trailing_atr_k": 1.2290405402496654}`
- Holdout metrics: trade_count=2, tpm=0.1644, pf_adj=1.3462, PF=10.0000, expectancy=298.2824, exp_lcb=282.5295, effective_edge=46.4432, exposure_ratio=0.0009, low_signal_penalty=0.9795, penalty_relief=False, max_dd=0.0005, return_pct=0.0298
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 8 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.2114282453293925, "atr_tp_multiplier": 1.445540427428266, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 55, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 24, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 25, "rsi_short_entry": 66, "trailing_atr_k": 1.8664674198625655}`
- Holdout metrics: trade_count=7, tpm=0.5753, pf_adj=2.1053, PF=10.0000, expectancy=96.7020, exp_lcb=77.7080, effective_edge=44.7087, exposure_ratio=0.0030, low_signal_penalty=0.9281, penalty_relief=False, max_dd=0.0005, return_pct=0.0338
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 9 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.7340717438372717, "atr_tp_multiplier": 4.499019129356344, "bollinger_period": 20, "bollinger_std": 2.0, "channel_period": 55, "ema_fast": 50, "ema_slow": 100, "max_holding_bars": 48, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 28, "rsi_short_entry": 69, "trailing_atr_k": 1.0244127739982085}`
- Holdout metrics: trade_count=4, tpm=0.3288, pf_adj=1.6667, PF=10.0000, expectancy=255.7230, exp_lcb=116.4312, effective_edge=38.2787, exposure_ratio=0.0050, low_signal_penalty=0.9589, penalty_relief=False, max_dd=0.0146, return_pct=0.0511
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 10 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `breakeven_1r`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.4542030447712078, "atr_tp_multiplier": 1.0574368161750725, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 100, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 24, "regime_gate_long": true, "regime_gate_short": true, "rsi_long_entry": 37, "rsi_short_entry": 64, "trailing_atr_k": 1.4885791063129088}`
- Holdout metrics: trade_count=24, tpm=1.9726, pf_adj=1.6196, PF=2.9106, expectancy=39.5008, exp_lcb=19.2792, effective_edge=38.0302, exposure_ratio=0.0099, low_signal_penalty=0.7534, penalty_relief=False, max_dd=0.0198, return_pct=0.0474
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 11 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `none`
- Exit mode: `fixed_atr`
- Holdout months used: `9`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.5257538029504538, "atr_tp_multiplier": 2.6419882504839536, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 55, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 48, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 36, "rsi_short_entry": 80, "trailing_atr_k": 1.8045079545810143}`
- Holdout metrics: trade_count=23, tpm=2.5000, pf_adj=1.4985, PF=2.5822, expectancy=39.0920, exp_lcb=14.6575, effective_edge=36.6439, exposure_ratio=0.0205, low_signal_penalty=0.6875, penalty_relief=False, max_dd=0.0579, return_pct=0.0450
- Holdout range: `2025-05-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 12 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.0749720144899566, "atr_tp_multiplier": 2.385971445252195, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 100, "ema_fast": 50, "ema_slow": 100, "max_holding_bars": 24, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 23, "rsi_short_entry": 66, "trailing_atr_k": 1.1858046622963316}`
- Holdout metrics: trade_count=7, tpm=0.5753, pf_adj=2.1053, PF=10.0000, expectancy=115.0062, exp_lcb=60.5106, effective_edge=34.8143, exposure_ratio=0.0047, low_signal_penalty=0.9281, penalty_relief=False, max_dd=0.0174, return_pct=0.0403
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 13 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 0.8833481075962414, "atr_tp_multiplier": 3.2342111686973203, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 55, "ema_fast": 50, "ema_slow": 100, "max_holding_bars": 96, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 24, "rsi_short_entry": 65, "trailing_atr_k": 1.9551625301575457}`
- Holdout metrics: trade_count=11, tpm=0.9041, pf_adj=1.4234, PF=3.3477, expectancy=110.9096, exp_lcb=37.8230, effective_edge=34.1962, exposure_ratio=0.0096, low_signal_penalty=0.8870, penalty_relief=False, max_dd=0.0323, return_pct=0.0610
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 14 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.3803182524121915, "atr_tp_multiplier": 1.4607442132570427, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 55, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 24, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 37, "rsi_short_entry": 66, "trailing_atr_k": 1.845793605206971}`
- Holdout metrics: trade_count=18, tpm=1.4795, pf_adj=1.5121, PF=2.9346, expectancy=47.4175, exp_lcb=22.4636, effective_edge=33.2338, exposure_ratio=0.0080, low_signal_penalty=0.8151, penalty_relief=False, max_dd=0.0225, return_pct=0.0427
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 15 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.2833113852489486, "atr_tp_multiplier": 3.2421489330660194, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 20, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 12, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 24, "rsi_short_entry": 70, "trailing_atr_k": 1.606758848750133}`
- Holdout metrics: trade_count=2, tpm=0.1644, pf_adj=1.3462, PF=10.0000, expectancy=250.8928, exp_lcb=201.6304, effective_edge=33.1447, exposure_ratio=0.0008, low_signal_penalty=0.9795, penalty_relief=False, max_dd=0.0006, return_pct=0.0251
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 16 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `breakeven_1r`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.7393367369055839, "atr_tp_multiplier": 4.143311836720374, "bollinger_period": 20, "bollinger_std": 2.0, "channel_period": 20, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 96, "regime_gate_long": false, "regime_gate_short": false, "rsi_long_entry": 40, "rsi_short_entry": 74, "trailing_atr_k": 2.488709421453188}`
- Holdout metrics: trade_count=26, tpm=2.1370, pf_adj=1.2845, PF=1.8316, expectancy=62.7307, exp_lcb=12.2663, effective_edge=26.2129, exposure_ratio=0.0218, low_signal_penalty=0.7329, penalty_relief=False, max_dd=0.0737, return_pct=0.0815
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 17 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.9818651992020315, "atr_tp_multiplier": 4.640358696520796, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 55, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 96, "regime_gate_long": false, "regime_gate_short": false, "rsi_long_entry": 22, "rsi_short_entry": 76, "trailing_atr_k": 1.2715147760052585}`
- Holdout metrics: trade_count=1, tpm=0.0822, pf_adj=1.1765, PF=10.0000, expectancy=265.8505, exp_lcb=265.8505, effective_edge=21.8507, exposure_ratio=0.0002, low_signal_penalty=0.9897, penalty_relief=False, max_dd=0.0003, return_pct=0.0133
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 18 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.147676407956741, "atr_tp_multiplier": 3.6592030620597913, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 100, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 24, "regime_gate_long": true, "regime_gate_short": true, "rsi_long_entry": 40, "rsi_short_entry": 73, "trailing_atr_k": 2.3413488748426987}`
- Holdout metrics: trade_count=27, tpm=2.2192, pf_adj=1.2357, PF=1.6723, expectancy=54.9005, exp_lcb=9.1897, effective_edge=20.3936, exposure_ratio=0.0237, low_signal_penalty=0.7226, penalty_relief=False, max_dd=0.0704, return_pct=0.0741
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 19 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.2562442829570384, "atr_tp_multiplier": 4.03407811934084, "bollinger_period": 20, "bollinger_std": 2.0, "channel_period": 55, "ema_fast": 50, "ema_slow": 100, "max_holding_bars": 12, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 26, "rsi_short_entry": 76, "trailing_atr_k": 2.0791944339264052}`
- Holdout metrics: trade_count=1, tpm=0.0822, pf_adj=1.1765, PF=10.0000, expectancy=229.1560, exp_lcb=229.1560, effective_edge=18.8347, exposure_ratio=0.0002, low_signal_penalty=0.9897, penalty_relief=False, max_dd=0.0003, return_pct=0.0115
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 20 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.2325296404154624, "atr_tp_multiplier": 3.5692219187737475, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 55, "ema_fast": 50, "ema_slow": 100, "max_holding_bars": 24, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 20, "rsi_short_entry": 76, "trailing_atr_k": 1.6478445197435425}`
- Holdout metrics: trade_count=1, tpm=0.0822, pf_adj=1.1765, PF=10.0000, expectancy=201.0210, exp_lcb=201.0210, effective_edge=16.5223, exposure_ratio=0.0001, low_signal_penalty=0.9897, penalty_relief=False, max_dd=0.0003, return_pct=0.0101
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 21 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `breakeven_1r`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.452491457695319, "atr_tp_multiplier": 1.5556156305969266, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 100, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 48, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 29, "rsi_short_entry": 65, "trailing_atr_k": 1.8602789698490751}`
- Holdout metrics: trade_count=11, tpm=0.9041, pf_adj=1.2382, PF=2.3211, expectancy=69.6364, exp_lcb=15.8476, effective_edge=14.3279, exposure_ratio=0.0059, low_signal_penalty=0.8870, penalty_relief=False, max_dd=0.0222, return_pct=0.0383
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 22 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `breakeven_1r`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.406494607420375, "atr_tp_multiplier": 1.2508183496127034, "bollinger_period": 20, "bollinger_std": 2.0, "channel_period": 55, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 12, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 31, "rsi_short_entry": 67, "trailing_atr_k": 1.1785919310734487}`
- Holdout metrics: trade_count=5, tpm=0.4110, pf_adj=1.8182, PF=10.0000, expectancy=54.2879, exp_lcb=24.7440, effective_edge=10.1688, exposure_ratio=0.0022, low_signal_penalty=0.9486, penalty_relief=False, max_dd=0.0012, return_pct=0.0136
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 23 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.5800351169804272, "atr_tp_multiplier": 4.405036104697668, "bollinger_period": 20, "bollinger_std": 1.5, "channel_period": 100, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 96, "regime_gate_long": false, "regime_gate_short": false, "rsi_long_entry": 40, "rsi_short_entry": 80, "trailing_atr_k": 1.4568945269920612}`
- Holdout metrics: trade_count=23, tpm=1.8904, pf_adj=1.2213, PF=1.7023, expectancy=62.8588, exp_lcb=4.9367, effective_edge=9.3324, exposure_ratio=0.0227, low_signal_penalty=0.7637, penalty_relief=False, max_dd=0.0729, return_pct=0.0723
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 24 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.1889197771757525, "atr_tp_multiplier": 4.053706140648487, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 55, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 12, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 34, "rsi_short_entry": 70, "trailing_atr_k": 1.4623356183524696}`
- Holdout metrics: trade_count=8, tpm=0.6575, pf_adj=2.2414, PF=10.0000, expectancy=61.8766, exp_lcb=11.5686, effective_edge=7.6067, exposure_ratio=0.0047, low_signal_penalty=0.9178, penalty_relief=False, max_dd=0.0228, return_pct=0.0248
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 25 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `trailing_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.3129132765310472, "atr_tp_multiplier": 2.947874491383849, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 55, "ema_fast": 50, "ema_slow": 100, "max_holding_bars": 96, "regime_gate_long": true, "regime_gate_short": true, "rsi_long_entry": 29, "rsi_short_entry": 76, "trailing_atr_k": 1.005465994295153}`
- Holdout metrics: trade_count=1, tpm=0.0822, pf_adj=1.1765, PF=10.0000, expectancy=76.2410, exp_lcb=76.2410, effective_edge=6.2664, exposure_ratio=0.0001, low_signal_penalty=0.9897, penalty_relief=False, max_dd=0.0003, return_pct=0.0038
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 26 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `partial_then_trail`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.9019123404341285, "atr_tp_multiplier": 4.12702819029578, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 20, "ema_fast": 20, "ema_slow": 200, "max_holding_bars": 96, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 29, "rsi_short_entry": 75, "trailing_atr_k": 1.64225479526331}`
- Holdout metrics: trade_count=1, tpm=0.0822, pf_adj=1.1765, PF=10.0000, expectancy=67.2167, exp_lcb=67.2167, effective_edge=5.5247, exposure_ratio=0.0001, low_signal_penalty=0.9897, penalty_relief=False, max_dd=0.0003, return_pct=0.0034
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 27 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol`
- Exit mode: `partial_then_trail`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.5045263608838395, "atr_tp_multiplier": 3.101132654290733, "bollinger_period": 20, "bollinger_std": 2.0, "channel_period": 20, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 48, "regime_gate_long": false, "regime_gate_short": false, "rsi_long_entry": 24, "rsi_short_entry": 75, "trailing_atr_k": 1.4765835022139406}`
- Holdout metrics: trade_count=1, tpm=0.0822, pf_adj=1.1765, PF=10.0000, expectancy=60.6436, exp_lcb=60.6436, effective_edge=4.9844, exposure_ratio=0.0001, low_signal_penalty=0.9897, penalty_relief=False, max_dd=0.0003, return_pct=0.0030
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 28 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `partial_then_trail`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 0.9431035222168322, "atr_tp_multiplier": 3.788766285496412, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 20, "ema_fast": 50, "ema_slow": 100, "max_holding_bars": 48, "regime_gate_long": true, "regime_gate_short": true, "rsi_long_entry": 20, "rsi_short_entry": 76, "trailing_atr_k": 2.4530173016746915}`
- Holdout metrics: trade_count=1, tpm=0.0822, pf_adj=1.1765, PF=10.0000, expectancy=13.5377, exp_lcb=13.5377, effective_edge=1.1127, exposure_ratio=0.0001, low_signal_penalty=0.9897, penalty_relief=False, max_dd=0.0003, return_pct=0.0007
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 29 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `vol+regime`
- Exit mode: `fixed_atr`
- Holdout months used: `12`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 2.1466677987301583, "atr_tp_multiplier": 2.740280577320264, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 20, "ema_fast": 50, "ema_slow": 100, "max_holding_bars": 12, "regime_gate_long": false, "regime_gate_short": true, "rsi_long_entry": 34, "rsi_short_entry": 71, "trailing_atr_k": 1.7470662514768256}`
- Holdout metrics: trade_count=8, tpm=0.6575, pf_adj=2.2414, PF=10.0000, expectancy=44.4570, exp_lcb=-3.1796, effective_edge=-2.0907, exposure_ratio=0.0035, low_signal_penalty=0.9178, penalty_relief=False, max_dd=0.0246, return_pct=0.0178
- Holdout range: `2025-02-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`

### Rank 30 - TrendPullback (Near Miss)
- Strategy: `Trend Pullback`
- Gating: `none`
- Exit mode: `fixed_atr`
- Holdout months used: `3`
- Entry rules: Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry.
- Exit rules: ATR stop loss, ATR take profit, or time stop from backtest engine.
- Parameters: `{"atr_sl_multiplier": 1.3843071308950183, "atr_tp_multiplier": 4.797011256423314, "bollinger_period": 20, "bollinger_std": 2.5, "channel_period": 100, "ema_fast": 50, "ema_slow": 200, "max_holding_bars": 24, "regime_gate_long": true, "regime_gate_short": false, "rsi_long_entry": 37, "rsi_short_entry": 60, "trailing_atr_k": 1.486186321902129}`
- Holdout metrics: trade_count=40, tpm=13.0435, pf_adj=1.1979, PF=1.4452, expectancy=29.0034, exp_lcb=-1.3060, effective_edge=-17.0351, exposure_ratio=0.1046, low_signal_penalty=0.0000, penalty_relief=False, max_dd=0.1495, return_pct=0.0580
- Holdout range: `2025-11-26T09:00:00+00:00..2026-02-26T09:00:00+00:00`
