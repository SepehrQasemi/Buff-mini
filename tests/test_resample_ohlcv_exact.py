from __future__ import annotations

import pandas as pd

from buffmini.data.resample import resample_ohlcv


def _fixture_1m_15rows() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=15, freq="1min", tz="UTC")
    open_ = [float(i + 1) for i in range(15)]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": [value + 1.0 for value in open_],
            "low": [value - 1.0 for value in open_],
            "close": [value + 0.5 for value in open_],
            "volume": [1.0 for _ in range(15)],
        }
    )


def test_resample_exact_5m() -> None:
    frame = _fixture_1m_15rows()
    out = resample_ohlcv(frame, target_timeframe="5m", base_timeframe="1m")
    assert len(out) == 3
    first = out.iloc[0]
    assert float(first["open"]) == 1.0
    assert float(first["high"]) == 6.0
    assert float(first["low"]) == 0.0
    assert float(first["close"]) == 5.5
    assert float(first["volume"]) == 5.0


def test_resample_exact_15m() -> None:
    frame = _fixture_1m_15rows()
    out = resample_ohlcv(frame, target_timeframe="15m", base_timeframe="1m")
    assert len(out) == 1
    row = out.iloc[0]
    assert float(row["open"]) == 1.0
    assert float(row["high"]) == 16.0
    assert float(row["low"]) == 0.0
    assert float(row["close"]) == 15.5
    assert float(row["volume"]) == 15.0

