"""Verify indicator computation avoids future leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.data.features import calculate_features


def test_features_are_stable_when_future_rows_are_appended() -> None:
    rng = np.random.default_rng(42)
    n = 260
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    high = close + rng.uniform(0.1, 1.0, size=n)
    low = close - rng.uniform(0.1, 1.0, size=n)

    base = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1000, 2000, size=n),
        }
    )

    base_features = calculate_features(base)

    extra_timestamps = pd.date_range(timestamps[-1] + pd.Timedelta(hours=1), periods=20, freq="h", tz="UTC")
    extra = pd.DataFrame(
        {
            "timestamp": extra_timestamps,
            "open": 1_000.0,
            "high": 2_000.0,
            "low": 100.0,
            "close": 1_500.0,
            "volume": 5_000.0,
        }
    )

    extended = pd.concat([base, extra], ignore_index=True)
    ext_features = calculate_features(extended).iloc[:n].reset_index(drop=True)

    columns = [
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "atr_14",
        "donchian_high_20",
        "donchian_low_20",
    ]

    pd.testing.assert_frame_equal(base_features[columns], ext_features[columns])
