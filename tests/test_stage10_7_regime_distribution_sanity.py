"""Stage-10.7 regime calibration sanity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.data.features import calculate_features


def _mixed_regime_ohlcv(rows: int = 1800) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")

    trend_seg = np.linspace(100.0, 170.0, 600) + rng.normal(0.0, 0.3, 600)
    range_base = 150.0 + 4.0 * np.sin(np.linspace(0.0, 18.0 * np.pi, 600))
    range_seg = range_base + rng.normal(0.0, 0.6, 600)
    vol_seg = 150.0 + np.cumsum(rng.normal(0.0, 2.5, 600))

    close = np.concatenate([trend_seg, range_seg, vol_seg])
    open_ = close + rng.normal(0.0, 0.4, rows)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.5, rows)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.5, rows)
    volume = np.concatenate(
        [
            rng.uniform(900.0, 1300.0, 600),
            rng.uniform(700.0, 1000.0, 600),
            rng.uniform(1300.0, 2200.0, 600),
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_regime_distribution_has_diversity_on_mixed_series() -> None:
    frame = _mixed_regime_ohlcv()
    features = calculate_features(frame)
    labels = features["regime_label_stage10"].astype(str)
    distribution = labels.value_counts(normalize=True) * 100.0
    rich = distribution[distribution > 10.0]
    assert len(rich) >= 3, f"distribution too concentrated: {distribution.to_dict()}"
