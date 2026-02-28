"""Stage-8.1 walk-forward v2 tests."""

from __future__ import annotations

import pandas as pd

from buffmini.baselines.stage0 import donchian_breakout
from buffmini.data.features import calculate_features
from buffmini.validation.walkforward_v2 import (
    WindowTriplet,
    aggregate_windows,
    build_windows,
    evaluate_candidate_on_window,
)


def _frame(hours: int = 24 * 80) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    close = pd.Series(range(hours), dtype=float) * 0.1 + 100.0
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0,
        }
    )


def test_build_windows_non_overlap_and_reserve_tail() -> None:
    windows = build_windows(
        start_ts="2024-01-01T00:00:00Z",
        end_ts="2024-06-01T00:00:00Z",
        train_days=30,
        holdout_days=10,
        forward_days=10,
        step_days=10,
        reserve_tail_days=20,
    )
    assert windows
    for left, right in zip(windows[:-1], windows[1:], strict=False):
        assert left.holdout_end <= left.forward_start
        assert left.forward_end <= right.holdout_start
    end = pd.Timestamp("2024-06-01T00:00:00Z")
    reserve_cutoff = end - pd.Timedelta(days=20)
    assert all(window.train_end <= reserve_cutoff for window in windows)


def test_aggregate_windows_classification() -> None:
    rows = [
        {
            "window_idx": 0,
            "usable": True,
            "exclude_reasons": "",
            "forward_expectancy": 1.0,
            "forward_profit_factor": 1.2,
            "forward_return_pct": 0.1,
            "forward_max_drawdown": 0.1,
        },
        {
            "window_idx": 1,
            "usable": True,
            "exclude_reasons": "",
            "forward_expectancy": 0.8,
            "forward_profit_factor": 1.1,
            "forward_return_pct": 0.05,
            "forward_max_drawdown": 0.08,
        },
        {
            "window_idx": 2,
            "usable": True,
            "exclude_reasons": "",
            "forward_expectancy": 0.6,
            "forward_profit_factor": 1.05,
            "forward_return_pct": 0.02,
            "forward_max_drawdown": 0.12,
        },
    ]
    cfg = {"evaluation": {"stage8": {"walkforward_v2": {"min_usable_windows": 3}}}}
    summary = aggregate_windows(rows, cfg)
    assert summary["classification"] == "STABLE"
    assert summary["usable_windows"] == 3


def test_evaluate_candidate_on_window_returns_finite() -> None:
    frame = calculate_features(_frame())
    window = WindowTriplet(
        window_idx=0,
        train_start=pd.Timestamp("2024-01-01T00:00:00Z"),
        train_end=pd.Timestamp("2024-02-01T00:00:00Z"),
        holdout_start=pd.Timestamp("2024-02-01T00:00:00Z"),
        holdout_end=pd.Timestamp("2024-02-15T00:00:00Z"),
        forward_start=pd.Timestamp("2024-02-15T00:00:00Z"),
        forward_end=pd.Timestamp("2024-03-01T00:00:00Z"),
    )
    result = evaluate_candidate_on_window(
        candidate={"strategy": donchian_breakout(), "symbol": "BTC/USDT", "gating_mode": "none"},
        data=frame,
        window_triplet=window,
        cfg={"costs": {"round_trip_cost_pct": 0.1, "slippage_pct": 0.0005}, "risk": {"max_concurrent_positions": 1}},
    )
    assert result["window_idx"] == 0
    assert isinstance(result["usable"], bool)
    for key in ["forward_expectancy", "forward_profit_factor", "forward_max_drawdown", "forward_return_pct"]:
        assert pd.notna(result[key])

