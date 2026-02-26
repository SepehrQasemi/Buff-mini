"""Cost model behavior tests."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from buffmini.backtest.costs import apply_fee, apply_slippage, round_trip_pct_to_one_way_fee_rate
from buffmini.backtest.engine import run_backtest


def _single_trade_frame(price_move: float = 2.0) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=20, freq="h", tz="UTC")
    close = [100.0] * 20
    high = [100.2] * 20
    low = [99.8] * 20

    close[10] = 100.0
    close[11] = 100.0 + float(price_move)

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


def _multi_trade_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=40, freq="h", tz="UTC")
    close = [100.0] * 40
    high = [100.3] * 40
    low = [99.7] * 40

    trade_setup = {
        10: 2.0,
        14: -1.0,
        18: 1.5,
        22: -0.5,
        26: 2.5,
        30: -1.25,
    }
    signal = [0] * 40
    for idx, move in trade_setup.items():
        close[idx] = 100.0
        close[idx + 1] = 100.0 + move
        signal[idx] = 1

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1_000.0] * 40,
            "atr_14": [1.0] * 40,
            "signal": signal,
        }
    )


def test_round_trip_percent_conversion_yields_expected_total_fee() -> None:
    one_way_rate = round_trip_pct_to_one_way_fee_rate(0.1)
    total_fee = apply_fee(1000.0, one_way_rate) + apply_fee(1000.0, one_way_rate)

    assert total_fee == pytest.approx(1.0)


def test_apply_slippage_is_directional() -> None:
    assert apply_slippage(price=100.0, slippage_pct=0.001, side="buy") == pytest.approx(100.1)
    assert apply_slippage(price=100.0, slippage_pct=0.001, side="sell") == pytest.approx(99.9)


def test_final_equity_is_higher_with_lower_round_trip_percent_cost() -> None:
    frame = _single_trade_frame(price_move=2.0)

    low_cost = run_backtest(
        frame=frame,
        strategy_name="CostTest",
        symbol="BTC/USDT",
        stop_atr_multiple=20.0,
        take_profit_atr_multiple=20.0,
        max_hold_bars=1,
        round_trip_cost_pct=0.1,
        slippage_pct=0.0,
    )
    high_cost = run_backtest(
        frame=frame,
        strategy_name="CostTest",
        symbol="BTC/USDT",
        stop_atr_multiple=20.0,
        take_profit_atr_multiple=20.0,
        max_hold_bars=1,
        round_trip_cost_pct=0.2,
        slippage_pct=0.0,
    )

    low_equity = float(low_cost.equity_curve["equity"].iloc[-1])
    high_equity = float(high_cost.equity_curve["equity"].iloc[-1])
    assert low_equity >= high_equity


def test_metrics_remain_finite_for_reasonable_percent_costs() -> None:
    frame = _multi_trade_frame()

    for cost in (0.05, 0.1, 0.2, 1.0):
        result = run_backtest(
            frame=frame,
            strategy_name="FiniteMetrics",
            symbol="BTC/USDT",
            stop_atr_multiple=20.0,
            take_profit_atr_multiple=20.0,
            max_hold_bars=1,
            round_trip_cost_pct=cost,
            slippage_pct=0.0,
        )

        assert int(result.metrics["trade_count"]) > 0
        for key, value in result.metrics.items():
            assert math.isfinite(float(value)), f"non-finite metric at cost={cost}: {key}={value}"
