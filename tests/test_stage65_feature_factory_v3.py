from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage65 import build_feature_frame_v3, compute_feature_attribution_v3


def test_stage65_builds_features_and_attribution() -> None:
    idx = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    close = 100.0 + np.cumsum(np.sin(np.arange(len(idx)) / 10.0) * 0.1)
    bars = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close + 0.05,
            "volume": 1000 + np.cos(np.arange(len(idx)) / 8.0) * 40,
        }
    )
    features, tags = build_feature_frame_v3(bars)
    label = (features["ret_1"] > 0).astype(int)
    importance = compute_feature_attribution_v3(features=features, label=label)
    assert not features.empty
    assert len(tags) > 0
    assert not importance.empty
    assert set(importance.columns) == {"feature", "importance", "method"}

