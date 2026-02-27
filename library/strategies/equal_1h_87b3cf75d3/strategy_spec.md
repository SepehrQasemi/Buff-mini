# Trading Spec

## 1) Objective & Scope
- This specification defines deterministic execution for the selected Buff-mini strategy portfolio.
- Instruments: `BTC/USDT, ETH/USDT`
- Timeframe: `1h`
- This document is for execution consistency and risk control, not a guarantee of profitability.

## 2) Inputs and Assumptions
- Stage-2 source run: `20260227_015806_3cb775eb81a0_stage2`
- Stage-1 source run: `20260227_010451_3ab7b86b23c2_stage1`
- Stage-3.3 source: `Stage-3.3`
- Costs: round_trip_cost_pct=0.1%, slippage_pct=0.0005
- Funding assumption: `0.0` per day
- Data source: local parquet cache under `data/raw/` loaded via project data layer.

## 3) Strategy Set (Portfolio)
- Selected method: `equal`
- Selected leverage: `5.0x`
- Execution mode: `net` (`per_symbol_netting=True`)
| candidate_id | family | gating | exit_mode | weight |
| --- | --- | --- | --- | ---: |
| cand_001106_0220dce2 | TrendPullback | none | fixed_atr | 0.142857 |
| cand_000637_42edc0b1 | TrendPullback | vol | fixed_atr | 0.142857 |
| cand_001264_848f658f | TrendPullback | vol | breakeven_1r | 0.142857 |
| cand_001139_d55f8520 | TrendPullback | vol | fixed_atr | 0.142857 |
| cand_001012_dbec7b03 | TrendPullback | none | fixed_atr | 0.142857 |
| cand_001603_418d4b75 | TrendPullback | vol+regime | breakeven_1r | 0.142857 |
| cand_001147_0131018a | TrendPullback | vol | fixed_atr | 0.142857 |

## 4) Entry Rules
- `cand_001106_0220dce2` (TrendPullback):
  code: `src/buffmini/baselines/stage0.py :: _trend_pullback`
  params: `{'channel_period': 100, 'ema_fast': 20, 'ema_slow': 200, 'rsi_long_entry': 35, 'rsi_short_entry': 64, 'bollinger_period': 20, 'bollinger_std': 2.0, 'atr_sl_multiplier': 2.267671495835672, 'atr_tp_multiplier': 4.136969411756131, 'trailing_atr_k': 2.0019281371341706, 'max_holding_bars': 48, 'regime_gate_long': False, 'regime_gate_short': True}`
  description: Trades pullbacks in EMA trend with RSI confirmation.
- `cand_000637_42edc0b1` (TrendPullback):
  code: `src/buffmini/baselines/stage0.py :: _trend_pullback`
  params: `{'channel_period': 55, 'ema_fast': 20, 'ema_slow': 200, 'rsi_long_entry': 35, 'rsi_short_entry': 64, 'bollinger_period': 20, 'bollinger_std': 2.5, 'atr_sl_multiplier': 1.2550028496017491, 'atr_tp_multiplier': 4.098810372044909, 'trailing_atr_k': 1.4967991102093166, 'max_holding_bars': 48, 'regime_gate_long': False, 'regime_gate_short': True}`
  description: Trades pullbacks in EMA trend with RSI confirmation.
- `cand_001264_848f658f` (TrendPullback):
  code: `src/buffmini/baselines/stage0.py :: _trend_pullback`
  params: `{'channel_period': 100, 'ema_fast': 50, 'ema_slow': 100, 'rsi_long_entry': 28, 'rsi_short_entry': 60, 'bollinger_period': 20, 'bollinger_std': 2.0, 'atr_sl_multiplier': 2.3998430132798356, 'atr_tp_multiplier': 4.158633815054241, 'trailing_atr_k': 1.0414705376315427, 'max_holding_bars': 24, 'regime_gate_long': True, 'regime_gate_short': True}`
  description: Trades pullbacks in EMA trend with RSI confirmation.
- `cand_001139_d55f8520` (TrendPullback):
  code: `src/buffmini/baselines/stage0.py :: _trend_pullback`
  params: `{'channel_period': 20, 'ema_fast': 50, 'ema_slow': 200, 'rsi_long_entry': 31, 'rsi_short_entry': 62, 'bollinger_period': 20, 'bollinger_std': 2.0, 'atr_sl_multiplier': 1.990928112487736, 'atr_tp_multiplier': 2.615986332726116, 'trailing_atr_k': 1.6090443409457977, 'max_holding_bars': 48, 'regime_gate_long': True, 'regime_gate_short': False}`
  description: Trades pullbacks in EMA trend with RSI confirmation.
- `cand_001012_dbec7b03` (TrendPullback):
  code: `src/buffmini/baselines/stage0.py :: _trend_pullback`
  params: `{'channel_period': 55, 'ema_fast': 20, 'ema_slow': 200, 'rsi_long_entry': 36, 'rsi_short_entry': 80, 'bollinger_period': 20, 'bollinger_std': 2.5, 'atr_sl_multiplier': 1.5257538029504538, 'atr_tp_multiplier': 2.6419882504839536, 'trailing_atr_k': 1.8045079545810143, 'max_holding_bars': 48, 'regime_gate_long': True, 'regime_gate_short': False}`
  description: Trades pullbacks in EMA trend with RSI confirmation.
- `cand_001603_418d4b75` (TrendPullback):
  code: `src/buffmini/baselines/stage0.py :: _trend_pullback`
  params: `{'channel_period': 20, 'ema_fast': 20, 'ema_slow': 200, 'rsi_long_entry': 40, 'rsi_short_entry': 74, 'bollinger_period': 20, 'bollinger_std': 2.0, 'atr_sl_multiplier': 1.7393367369055839, 'atr_tp_multiplier': 4.143311836720374, 'trailing_atr_k': 2.488709421453188, 'max_holding_bars': 96, 'regime_gate_long': False, 'regime_gate_short': False}`
  description: Trades pullbacks in EMA trend with RSI confirmation.
- `cand_001147_0131018a` (TrendPullback):
  code: `src/buffmini/baselines/stage0.py :: _trend_pullback`
  params: `{'channel_period': 100, 'ema_fast': 50, 'ema_slow': 200, 'rsi_long_entry': 40, 'rsi_short_entry': 73, 'bollinger_period': 20, 'bollinger_std': 2.5, 'atr_sl_multiplier': 2.147676407956741, 'atr_tp_multiplier': 3.6592030620597913, 'trailing_atr_k': 2.3413488748426987, 'max_holding_bars': 24, 'regime_gate_long': True, 'regime_gate_short': True}`
  description: Trades pullbacks in EMA trend with RSI confirmation.

## 5) Exit Rules
- Exit implementation: `src/buffmini/backtest/engine.py :: run_backtest`
- Uses ATR stop, ATR target, and time-stop with configured exit mode per candidate.
- Precedence on same candle follows stop-first conservative execution.
- Gating filters (volatility/regime) are applied at signal generation stage.

## 6) Position Sizing & Leverage
- Leverage: `5.0x` (Stage-3.3)
- sizing.mode: `risk_budget`
- sizing.risk_per_trade_pct: `1.0%`
- sizing.fixed_fraction_pct: `10.0%`
- max_gross_exposure: `5.0x`
- max_net_exposure_per_symbol: `5.0x`
- On cap breach, all desired exposures are scaled by one multiplier to remain inside limits.

## 7) Kill-Switch
- enabled: `True`
- max_daily_loss_pct: `5.0%`
- max_peak_to_valley_dd_pct: `20.0%`
- max_consecutive_losses: `8`
- cool_down_bars: `48`
- Trigger behavior: stop opening new positions until cooldown expires; existing positions are not force-closed.

## 8) Re-Evaluation Plan
- cadence: `weekly`
- min_new_bars: `168`
- Re-run Stage-1 discovery, Stage-2 portfolio build, and Stage-3.3 leverage selection on schedule.

## 9) Monitoring Checklist (Live)
- Log per bar: equity, gross exposure, per-symbol net exposure, open positions, cooldown state.
- Log per day: return, drawdown, loss streak, cap-bind events, kill-switch events.
- Alert when: cap scaling > 0, kill-switch triggers, exposure exceeds configured limits.

No guarantee of profitability. Designed to reduce overfitting and execution drift.
