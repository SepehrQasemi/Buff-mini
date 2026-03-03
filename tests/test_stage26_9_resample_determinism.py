from __future__ import annotations

import pandas as pd

from buffmini.data.resample import resample_monthly_ohlcv, resample_ohlcv
from buffmini.utils.hashing import stable_hash


def _frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="list")
    return stable_hash(payload, length=24)


def test_stage26_9_resample_determinism_intraday() -> None:
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=500, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0 + float(i) * 0.1 for i in range(len(ts))],
            "high": [100.5 + float(i) * 0.1 for i in range(len(ts))],
            "low": [99.5 + float(i) * 0.1 for i in range(len(ts))],
            "close": [100.2 + float(i) * 0.1 for i in range(len(ts))],
            "volume": [1.0 + float(i % 5) for i in range(len(ts))],
        }
    )
    a = resample_ohlcv(frame, target_timeframe="30m", base_timeframe="1m", partial_last_bucket=False)
    b = resample_ohlcv(frame, target_timeframe="30m", base_timeframe="1m", partial_last_bucket=False)
    assert _frame_hash(a) == _frame_hash(b)


def test_stage26_9_resample_determinism_monthly() -> None:
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=90 * 24 * 60, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [200.0 for _ in range(len(ts))],
            "high": [201.0 for _ in range(len(ts))],
            "low": [199.0 for _ in range(len(ts))],
            "close": [200.5 for _ in range(len(ts))],
            "volume": [2.0 for _ in range(len(ts))],
        }
    )
    a = resample_monthly_ohlcv(frame, partial_last_bucket=False)
    b = resample_monthly_ohlcv(frame, partial_last_bucket=False)
    assert _frame_hash(a) == _frame_hash(b)
