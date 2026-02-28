"""Stage-8.2 cost model v2 tests."""

from __future__ import annotations

import math

import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.validation.cost_model_v2 import (
    normalize_cost_model_cfg,
    one_way_cost_breakdown_bps,
)


def _frame(rows: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = [100.0 + (idx * 0.1) for idx in range(rows)]
    high = [value + (0.5 if idx < rows // 2 else 2.5) for idx, value in enumerate(close)]
    low = [value - (0.5 if idx < rows // 2 else 2.5) for idx, value in enumerate(close)]
    signal = [0] * rows
    for idx in range(10, rows - 5, 12):
        signal[idx] = 1
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0,
            "atr_14": [1.0 if idx < rows // 2 else 3.0 for idx in range(rows)],
            "signal": signal,
        }
    )
    return frame


def test_v2_costs_are_not_lower_than_simple_for_positive_spread_slippage() -> None:
    frame = _frame()
    simple = run_backtest(
        frame=frame,
        strategy_name="CostSimple",
        symbol="BTC/USDT",
        max_hold_bars=1,
        stop_atr_multiple=20.0,
        take_profit_atr_multiple=20.0,
        round_trip_cost_pct=0.1,
        slippage_pct=0.0002,
    )
    v2 = run_backtest(
        frame=frame,
        strategy_name="CostV2",
        symbol="BTC/USDT",
        max_hold_bars=1,
        stop_atr_multiple=20.0,
        take_profit_atr_multiple=20.0,
        round_trip_cost_pct=0.1,
        slippage_pct=0.0002,
        cost_model_cfg={
            "mode": "v2",
            "round_trip_cost_pct": 0.1,
            "v2": {
                "slippage_bps_base": 1.0,
                "slippage_bps_vol_mult": 2.0,
                "spread_bps": 1.0,
                "delay_bars": 0,
                "vol_proxy": "atr_pct",
                "vol_lookback": 14,
                "max_total_bps_per_side": 20.0,
            },
        },
    )
    simple_equity = float(simple.equity_curve["equity"].iloc[-1])
    v2_equity = float(v2.equity_curve["equity"].iloc[-1])
    assert v2_equity <= simple_equity


def test_increasing_volatility_increases_dynamic_slippage() -> None:
    frame = _frame()
    cfg = normalize_cost_model_cfg(
        cost_model_cfg={
            "mode": "v2",
            "round_trip_cost_pct": 0.1,
            "v2": {
                "slippage_bps_base": 0.5,
                "slippage_bps_vol_mult": 3.0,
                "spread_bps": 0.5,
                "delay_bars": 0,
                "vol_proxy": "atr_pct",
                "vol_lookback": 14,
                "max_total_bps_per_side": 500.0,
            },
        },
        round_trip_cost_pct=0.1,
        slippage_pct=0.0002,
    )
    low = one_way_cost_breakdown_bps(frame=frame, bar_index=20, cost_cfg=cfg)
    high = one_way_cost_breakdown_bps(frame=frame, bar_index=70, cost_cfg=cfg)
    assert high["dynamic_slippage_bps"] > low["dynamic_slippage_bps"]
    assert high["total_bps"] > low["total_bps"]


def test_delay_bars_shifts_fill_timestamps_deterministically() -> None:
    frame = _frame(rows=60)
    delay0 = run_backtest(
        frame=frame,
        strategy_name="Delay0",
        symbol="BTC/USDT",
        max_hold_bars=1,
        stop_atr_multiple=20.0,
        take_profit_atr_multiple=20.0,
        round_trip_cost_pct=0.1,
        slippage_pct=0.0,
        cost_model_cfg={
            "mode": "v2",
            "round_trip_cost_pct": 0.1,
            "v2": {
                "slippage_bps_base": 0.0,
                "slippage_bps_vol_mult": 0.0,
                "spread_bps": 0.0,
                "delay_bars": 0,
                "vol_proxy": "atr_pct",
                "vol_lookback": 14,
                "max_total_bps_per_side": 10.0,
            },
        },
    )
    delay1 = run_backtest(
        frame=frame,
        strategy_name="Delay1",
        symbol="BTC/USDT",
        max_hold_bars=1,
        stop_atr_multiple=20.0,
        take_profit_atr_multiple=20.0,
        round_trip_cost_pct=0.1,
        slippage_pct=0.0,
        cost_model_cfg={
            "mode": "v2",
            "round_trip_cost_pct": 0.1,
            "v2": {
                "slippage_bps_base": 0.0,
                "slippage_bps_vol_mult": 0.0,
                "spread_bps": 0.0,
                "delay_bars": 1,
                "vol_proxy": "atr_pct",
                "vol_lookback": 14,
                "max_total_bps_per_side": 10.0,
            },
        },
    )
    assert not delay0.trades.empty and not delay1.trades.empty
    first0 = pd.to_datetime(delay0.trades.iloc[0]["entry_time"], utc=True)
    first1 = pd.to_datetime(delay1.trades.iloc[0]["entry_time"], utc=True)
    assert first1 == first0 + pd.Timedelta(hours=1)


def test_v2_metrics_are_finite() -> None:
    frame = _frame()
    result = run_backtest(
        frame=frame,
        strategy_name="FiniteV2",
        symbol="BTC/USDT",
        max_hold_bars=1,
        stop_atr_multiple=20.0,
        take_profit_atr_multiple=20.0,
        round_trip_cost_pct=0.1,
        slippage_pct=0.0002,
        cost_model_cfg={
            "mode": "v2",
            "round_trip_cost_pct": 0.1,
            "v2": {
                "slippage_bps_base": 0.5,
                "slippage_bps_vol_mult": 2.0,
                "spread_bps": 0.5,
                "delay_bars": 1,
                "vol_proxy": "atr_pct",
                "vol_lookback": 14,
                "max_total_bps_per_side": 10.0,
            },
        },
    )
    for value in result.metrics.values():
        assert math.isfinite(float(value))
