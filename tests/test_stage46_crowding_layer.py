from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage46.part2 import compute_crowding_layer


def test_stage46_crowding_layer_enforces_oi_short_only_guard() -> None:
    idx = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0 + np.sin(np.arange(48) / 4.0),
            "volume": 1000.0,
            "funding_rate": np.linspace(-0.001, 0.001, 48),
            "long_short_ratio": np.linspace(0.8, 1.2, 48),
            "oi": np.linspace(10000, 12000, 48),
        }
    )
    out, guard = compute_crowding_layer(
        frame,
        timeframe="1h",
        short_only_enabled=True,
        short_horizon_max="30m",
    )
    assert bool(guard["oi_allowed"]) is False
    assert int(out["oi_active"].sum()) == 0

