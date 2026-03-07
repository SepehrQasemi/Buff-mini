from __future__ import annotations

import pandas as pd

from buffmini.stage45.part1 import compute_liquidity_map


def test_stage45_liquidity_map_detects_fake_breakout() -> None:
    idx = pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC")
    high = pd.Series([100, 101, 101.0, 101.01, 102.5, 101.0, 100.8, 100.7], dtype=float)
    low = pd.Series([99, 99.5, 99.4, 99.3, 99.2, 99.1, 99.0, 98.9], dtype=float)
    close = pd.Series([99.8, 100.5, 100.7, 100.6, 100.9, 100.4, 100.2, 100.0], dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": high,
            "low": low,
            "close": close,
            "volume": 800.0,
        }
    )
    out = compute_liquidity_map(frame)
    assert int(out["liquidity_pool_high"].sum()) >= 1
    assert int(out["fake_breakout"].sum()) >= 1

