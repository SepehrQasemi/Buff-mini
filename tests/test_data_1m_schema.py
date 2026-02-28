from __future__ import annotations

import pandas as pd
import pytest

from buffmini.data.loader import standardize_ohlcv_frame, validate_ohlcv_frame


def test_1m_schema_validation_accepts_clean_frame() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=12, freq="1min", tz="UTC"),
            "open": [100 + i for i in range(12)],
            "high": [101 + i for i in range(12)],
            "low": [99 + i for i in range(12)],
            "close": [100.5 + i for i in range(12)],
            "volume": [10.0 for _ in range(12)],
        }
    )
    validate_ohlcv_frame(frame)


def test_1m_schema_standardize_deduplicates_and_sorts() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:02:00Z",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:01:00Z",
                "2026-01-01T00:01:00Z",
            ],
            "open": [3.0, 1.0, 2.0, 2.1],
            "high": [3.1, 1.1, 2.1, 2.2],
            "low": [2.9, 0.9, 1.9, 2.0],
            "close": [3.0, 1.0, 2.0, 2.1],
            "volume": [1.0, 1.0, 1.0, 1.5],
        }
    )
    normalized = standardize_ohlcv_frame(frame)
    validate_ohlcv_frame(normalized)
    assert len(normalized) == 3
    assert normalized["timestamp"].is_monotonic_increasing


def test_1m_schema_validation_rejects_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "open": [1.0, 1.0],
            "high": [1.1, 1.1],
            "low": [0.9, 0.9],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="duplicates"):
        validate_ohlcv_frame(frame)

