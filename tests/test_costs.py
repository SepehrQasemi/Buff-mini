"""Cost model behavior tests."""

from __future__ import annotations

import pandas as pd

from buffmini.backtest.costs import apply_fee, apply_slippage
from buffmini.backtest.engine import run_backtest


def _cost_test_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=20, freq="h", tz="UTC")
    close = [100.0] * 20
    high = [100.0] * 20
    low = [100.0] * 20

    close[10] = 100.0
    close[11] = 102.0
    high[11] = 102.0
    low[11] = 100.0

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1_000.0] * 20,
            "atr_14": [1.0] * 20,
            "signal": [0] * 20,
        }
    )
    frame.loc[10, "signal"] = 1
    return frame


def test_apply_fee_uses_round_trip_pct_once_on_notional() -> None:
    assert apply_fee(notional=10_000.0, round_trip_cost_pct=0.001) == 10.0


def test_apply_slippage_is_directional() -> None:
    assert apply_slippage(price=100.0, slippage_pct=0.001, side="buy") == 100.1
    assert apply_slippage(price=100.0, slippage_pct=0.001, side="sell") == 99.9


def test_costs_reduce_pnl_for_identical_price_path() -> None:
    frame = _cost_test_frame()

    no_costs = run_backtest(
        frame=frame,
        strategy_name="CostTest",
        symbol="BTC/USDT",
        stop_atr_multiple=10.0,
        take_profit_atr_multiple=10.0,
        max_hold_bars=1,
        round_trip_cost_pct=0.0,
        slippage_pct=0.0,
    )
    with_costs = run_backtest(
        frame=frame,
        strategy_name="CostTest",
        symbol="BTC/USDT",
        stop_atr_multiple=10.0,
        take_profit_atr_multiple=10.0,
        max_hold_bars=1,
        round_trip_cost_pct=0.002,
        slippage_pct=0.001,
    )

    assert no_costs.trades.iloc[0]["pnl"] > with_costs.trades.iloc[0]["pnl"]
