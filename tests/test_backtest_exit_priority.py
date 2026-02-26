"""Exit-priority tests for deterministic stop-vs-target behavior."""

from __future__ import annotations

import pandas as pd

from buffmini.backtest.engine import run_backtest


def test_same_candle_hit_of_stop_and_target_uses_stop_first() -> None:
    timestamps = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
    close = [100.0] * 30
    high = [100.5] * 30
    low = [99.5] * 30

    # Entry at index 10, then both stop and TP are touched at index 11.
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1_000.0] * 30,
            "atr_14": [1.0] * 30,
            "signal": [0] * 30,
        }
    )
    frame.loc[10, "signal"] = 1
    frame.loc[11, "high"] = 102.0
    frame.loc[11, "low"] = 98.0

    result = run_backtest(
        frame=frame,
        strategy_name="PriorityTest",
        symbol="BTC/USDT",
        stop_atr_multiple=1.0,
        take_profit_atr_multiple=1.0,
        max_hold_bars=10,
        round_trip_cost_pct=0.0,
        slippage_pct=0.0,
    )

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "stop_loss"
