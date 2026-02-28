from __future__ import annotations

import pandas as pd

from buffmini.data.resample import assert_resample_is_causal, resample_ohlcv


def test_resample_causality_with_future_spike() -> None:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=10, freq="1min", tz="UTC")
    high = [100.0, 101.0, 102.0, 101.5, 103.0, 9999.0, 106.0, 107.0, 108.0, 109.0]
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0 + i for i in range(10)],
            "high": high,
            "low": [99.0 + i for i in range(10)],
            "close": [100.2 + i for i in range(10)],
            "volume": [1.0 for _ in range(10)],
        }
    )

    out = resample_ohlcv(frame, target_timeframe="5m", base_timeframe="1m")
    assert len(out) == 2
    # first bucket (00:00..00:04) must not include future spike at 00:05
    assert float(out.iloc[0]["high"]) == 103.0
    assert float(out.iloc[1]["high"]) == 9999.0
    assert_resample_is_causal(
        base_frame=frame,
        resampled_frame=out,
        target_timeframe="5m",
        base_timeframe="1m",
    )

