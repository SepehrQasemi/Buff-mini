from __future__ import annotations

import pandas as pd

from buffmini.mtf.resample import resample_ohlcv, validate_resampled_schema


def test_resample_schema_and_ohlcv_correctness() -> None:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="1h", tz="UTC")
    base = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1, 2, 3, 4, 5, 6, 7, 8],
            "high": [2, 3, 4, 5, 6, 7, 8, 9],
            "low": [0, 1, 2, 3, 4, 5, 6, 7],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
            "volume": [10, 10, 10, 10, 10, 10, 10, 10],
        }
    )

    out = resample_ohlcv(base_df=base, target_timeframe="4h")
    validate_resampled_schema(out)

    assert len(out) == 2
    assert list(out["open"]) == [1, 5]
    assert list(out["close"]) == [4.5, 8.5]
    assert list(out["high"]) == [5, 9]
    assert list(out["low"]) == [0, 4]
    assert list(out["volume"]) == [40, 40]
    assert out["ts_open"].iloc[0].isoformat() == "2026-01-01T00:00:00+00:00"
    assert out["ts_close"].iloc[0].isoformat() == "2026-01-01T04:00:00+00:00"

