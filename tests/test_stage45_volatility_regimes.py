from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage45.part1 import compute_volatility_regime_engine


def test_stage45_volatility_regime_detects_compression_and_expansion() -> None:
    idx = pd.date_range("2026-01-01", periods=80, freq="1h", tz="UTC")
    close = pd.Series(np.r_[np.linspace(100, 101, 40), np.linspace(101, 110, 40)], dtype=float)
    high = close + np.r_[np.full(40, 0.05), np.full(40, 1.0)]
    low = close - np.r_[np.full(40, 0.05), np.full(40, 1.0)]
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0,
        }
    )
    out = compute_volatility_regime_engine(frame)
    assert int(out["volatility_compression"].sum()) > 0
    assert int(out["volatility_expansion"].sum()) > 0

