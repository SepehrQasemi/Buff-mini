"""Basic backtest engine behavior tests."""

from __future__ import annotations

import pandas as pd

from buffmini.backtest.engine import run_backtest


def test_stop_loss_triggers_on_synthetic_series() -> None:
    timestamps = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
    close = [100.0] * 30
    high = [101.0] * 30
    low = [99.0] * 30

    # Entry bar and next-bar stop-out setup.
    close[15] = 100.0
    high[16] = 100.2
    low[16] = 98.8

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
    frame.loc[15, "signal"] = 1

    result = run_backtest(
        frame=frame,
        strategy_name="Test",
        symbol="BTC/USDT",
        stop_atr_multiple=1.0,
        take_profit_atr_multiple=3.0,
        max_hold_bars=10,
        round_trip_cost_pct=0.0,
        slippage_pct=0.0,
    )

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "stop_loss"
    assert result.metrics["trade_count"] == 1.0
