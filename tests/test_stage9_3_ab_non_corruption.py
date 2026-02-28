"""Stage-9.3 AB non-corruption guarantees for non-OI strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, trend_pullback
from buffmini.data.features import calculate_features
from buffmini.validation.leakage_harness import synthetic_ohlcv


def _overlay_config() -> dict:
    return {
        "data": {
            "include_futures_extras": True,
            "futures_extras": {
                "timeframe": "1h",
                "max_fill_gap_bars": 8,
                "funding": {"z_windows": [30, 90], "trend_window": 24, "abs_pctl_window": 180, "extreme_pctl": 0.95},
                "open_interest": {
                    "chg_windows": [1, 24],
                    "z_window": 30,
                    "oi_to_volume_window": 24,
                    "overlay": {
                        "enabled": True,
                        "recent_window_days": 30,
                        "max_recent_window_days": 90,
                        "clamp_to_available": True,
                        "inactive_value": "nan",
                    },
                },
            },
        },
        "universe": {"resolved_end_ts": "2026-02-28T00:00:00Z"},
    }


def test_ab_non_corruption_for_non_oi_strategy() -> None:
    raw = synthetic_ohlcv(rows=900, seed=123)

    base_features = calculate_features(raw)
    overlay_features = calculate_features(
        raw,
        config=_overlay_config(),
        symbol="BTC/USDT",
        timeframe="1h",
        _synthetic_extras_for_tests=True,
    )

    strategy = trend_pullback()
    base_df = base_features.copy()
    base_df["signal"] = generate_signals(base_df, strategy, gating_mode="none")

    overlay_df = overlay_features.copy()
    overlay_df["signal"] = generate_signals(overlay_df, strategy, gating_mode="none")

    result_a = run_backtest(
        frame=base_df,
        strategy_name=strategy.name,
        symbol="BTC/USDT",
        stop_atr_multiple=1.5,
        take_profit_atr_multiple=3.0,
        max_hold_bars=24,
        round_trip_cost_pct=0.1,
        slippage_pct=0.0005,
    )
    result_b = run_backtest(
        frame=overlay_df,
        strategy_name=strategy.name,
        symbol="BTC/USDT",
        stop_atr_multiple=1.5,
        take_profit_atr_multiple=3.0,
        max_hold_bars=24,
        round_trip_cost_pct=0.1,
        slippage_pct=0.0005,
    )

    trades_a = float(result_a.metrics["trade_count"])
    trades_b = float(result_b.metrics["trade_count"])
    if trades_a == 0:
        trade_count_delta_pct = 0.0
    else:
        trade_count_delta_pct = abs(trades_b - trades_a) / trades_a * 100.0
    assert trade_count_delta_pct <= 1.0

    eq_a = result_a.equity_curve["equity"].to_numpy(dtype=float)
    eq_b = result_b.equity_curve["equity"].to_numpy(dtype=float)
    assert eq_a.shape == eq_b.shape
    assert np.allclose(eq_a, eq_b, atol=1e-12, rtol=0.0)

