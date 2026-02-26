"""Stage-0.5 filter behavior tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, rsi_mean_reversion, stage0_strategies
from buffmini.data.features import calculate_features


def _make_filter_test_frame(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")

    base = 100 + np.sin(np.linspace(0, 40, rows)) * 3
    vol_state = np.where(np.arange(rows) % 80 < 40, 0.2, 2.0)
    close = base + rng.normal(0, vol_state, rows)
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    candle_spread = np.abs(rng.normal(0.6, 0.2, rows)) * vol_state
    high = np.maximum(open_, close) + candle_spread
    low = np.minimum(open_, close) - candle_spread

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1_000, 2_000, rows),
        }
    )


def test_stage05_filters_reduce_trade_count() -> None:
    frame = _make_filter_test_frame()
    features = calculate_features(frame)
    strategy = rsi_mean_reversion()

    unfiltered = features.copy()
    unfiltered["signal"] = generate_signals(unfiltered, strategy, stage05=False)
    unfiltered_result = run_backtest(
        frame=unfiltered,
        strategy_name=strategy.name,
        symbol="BTC/USDT",
        round_trip_cost_pct=0.1,
        slippage_pct=0.0,
    )

    filtered = features.copy()
    filtered["signal"] = generate_signals(filtered, strategy, stage05=True)
    filtered_result = run_backtest(
        frame=filtered,
        strategy_name=strategy.name,
        symbol="BTC/USDT",
        round_trip_cost_pct=0.1,
        slippage_pct=0.0,
    )

    assert filtered_result.metrics["trade_count"] < unfiltered_result.metrics["trade_count"]


@pytest.mark.parametrize("strategy_name", [s.name for s in stage0_strategies()])
def test_stage05_signal_generation_has_no_future_leakage(strategy_name: str) -> None:
    frame = _make_filter_test_frame(rows=520)

    strategy = next(s for s in stage0_strategies() if s.name == strategy_name)

    features = calculate_features(frame)
    baseline_signal = generate_signals(features, strategy, stage05=True)

    cutoff = 340
    shocked = frame.copy()
    shocked.loc[cutoff + 1 :, "close"] = shocked.loc[cutoff + 1 :, "close"] * 2.0
    shocked.loc[cutoff + 1 :, "high"] = shocked.loc[cutoff + 1 :, "high"] * 2.5
    shocked.loc[cutoff + 1 :, "low"] = shocked.loc[cutoff + 1 :, "low"] * 0.5

    shocked_features = calculate_features(shocked)
    shocked_signal = generate_signals(shocked_features, strategy, stage05=True)

    pd.testing.assert_series_equal(
        baseline_signal.loc[:cutoff].reset_index(drop=True),
        shocked_signal.loc[:cutoff].reset_index(drop=True),
        check_names=False,
    )
