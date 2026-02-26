"""Verify indicator computation avoids future leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buffmini.data.features import calculate_features


FEATURE_COLUMNS = [
    "ema_20",
    "ema_50",
    "ema_200",
    "rsi_14",
    "atr_14",
    "atr_14_sma_50",
    "donchian_high_20",
    "donchian_low_20",
]


def _make_frame(rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 1, size=rows))
    high = close + rng.uniform(0.1, 1.0, size=rows)
    low = close - rng.uniform(0.1, 1.0, size=rows)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1000, 2000, size=rows),
        }
    )


@pytest.mark.parametrize("feature_col", FEATURE_COLUMNS)
def test_future_shock_does_not_change_past_feature_values(feature_col: str) -> None:
    base = _make_frame()
    baseline = calculate_features(base)

    cutoff = 220
    shocked = base.copy()
    shocked.loc[cutoff + 1 :, "close"] = shocked.loc[cutoff + 1 :, "close"] * 25.0
    shocked.loc[cutoff + 1 :, "high"] = shocked.loc[cutoff + 1 :, "high"] * 30.0
    shocked.loc[cutoff + 1 :, "low"] = shocked.loc[cutoff + 1 :, "low"] * 0.2

    recomputed = calculate_features(shocked)

    pd.testing.assert_series_equal(
        baseline.loc[:cutoff, feature_col].reset_index(drop=True),
        recomputed.loc[:cutoff, feature_col].reset_index(drop=True),
        check_names=False,
    )
