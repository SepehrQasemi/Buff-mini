from __future__ import annotations

import pandas as pd

from buffmini.data.resample import resample_ohlcv


def _make_1m(rows: int = 120) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=rows, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [float(i) for i in range(rows)],
            "high": [float(i) + 0.5 for i in range(rows)],
            "low": [float(i) - 0.5 for i in range(rows)],
            "close": [float(i) + 0.25 for i in range(rows)],
            "volume": [1.0 for _ in range(rows)],
        }
    )


def test_stage26_9_resample_30m_exact() -> None:
    frame = _make_1m(60)
    out = resample_ohlcv(frame, target_timeframe="30m", base_timeframe="1m", partial_last_bucket=False)
    assert out.shape[0] == 2

    first = out.iloc[0]
    assert float(first["open"]) == 0.0
    assert float(first["high"]) == 29.5
    assert float(first["low"]) == -0.5
    assert float(first["close"]) == 29.25
    assert float(first["volume"]) == 30.0


def test_stage26_9_resample_2h_from_1m_exact() -> None:
    frame = _make_1m(240)
    out = resample_ohlcv(frame, target_timeframe="2h", base_timeframe="1m", partial_last_bucket=False)
    assert out.shape[0] == 2

    second = out.iloc[1]
    assert float(second["open"]) == 120.0
    assert float(second["high"]) == 239.5
    assert float(second["low"]) == 119.5
    assert float(second["close"]) == 239.25
    assert float(second["volume"]) == 120.0


def test_stage26_9_resample_6h_and_12h_counts() -> None:
    frame = _make_1m(24 * 60)
    out_6h = resample_ohlcv(frame, target_timeframe="6h", base_timeframe="1m", partial_last_bucket=False)
    out_12h = resample_ohlcv(frame, target_timeframe="12h", base_timeframe="1m", partial_last_bucket=False)
    assert out_6h.shape[0] == 4
    assert out_12h.shape[0] == 2
