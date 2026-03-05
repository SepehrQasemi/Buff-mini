from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.data.stage35_timeframes import build_timeframe_from_base, choose_resample_base, integrity_checks


def _base_1m(rows: int = 64_800) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=rows, freq="1min", tz="UTC")
    idx = np.arange(rows, dtype=float)
    price = 100.0 + np.sin(idx / 500.0) + (idx * 0.0002)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": price,
            "high": price + 0.3,
            "low": price - 0.3,
            "close": price + 0.05,
            "volume": 1000.0 + (idx % 7.0),
        }
    )


def test_choose_resample_base_prefers_1m_when_available() -> None:
    available = ["1m", "5m", "1h", "4h"]
    assert choose_resample_base(available, "30m") == "1m"
    assert choose_resample_base(available, "2h") == "1m"
    assert choose_resample_base(available, "1h") == "1h"


def test_stage35_resample_integrity_across_derived_timeframes() -> None:
    base = _base_1m()
    targets = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1M"]
    for target in targets:
        rebuilt = build_timeframe_from_base(base, base_timeframe="1m", target_timeframe=target, drop_incomplete_last=True)
        report = integrity_checks(base, rebuilt, base_timeframe="1m", target_timeframe=target, volume_tol=1e-6)
        assert report["high_low_consistent"] is True, target
        assert report["volume_consistent"] is True, target

