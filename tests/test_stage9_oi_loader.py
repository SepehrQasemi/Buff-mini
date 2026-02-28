"""Stage-9.2 open-interest loader tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.data.derived_store import load_derived_parquet, save_derived_parquet
from buffmini.data.futures_extras import align_open_interest_to_ohlcv, open_interest_quality_report


def _ohlcv() -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
            "volume": [1, 1, 1, 1, 1, 1],
        }
    )


def test_open_interest_alignment_latest_event_le_candle_close() -> None:
    ohlcv = _ohlcv()
    oi = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01T01:00:00Z",
                    "2025-01-01T03:30:00Z",
                ],
                utc=True,
            ),
            "open_interest": [1000.0, 1100.0],
        }
    )

    aligned = align_open_interest_to_ohlcv(ohlcv=ohlcv, open_interest=oi, timeframe="1h")

    assert float(aligned.loc[0, "open_interest"]) == 1000.0
    assert float(aligned.loc[2, "open_interest"]) == 1000.0
    assert float(aligned.loc[3, "open_interest"]) == 1100.0


def test_open_interest_quality_report_monotonic_and_duplicates() -> None:
    oi = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T01:00:00Z",
                    "2025-01-01T01:00:00Z",
                    "2025-01-01T03:30:00Z",
                ],
                utc=True,
            ),
            "open_interest": [10, 11, 12, 13],
        }
    )

    quality = open_interest_quality_report(oi)
    assert quality["rows"] == 3
    assert quality["monotonic_ts"] is True
    assert quality["duplicates"] == 1
    assert quality["gaps_count"] == 1


def test_open_interest_derived_roundtrip(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
            "open_interest": [1000.0, 1005.0, 1010.0, 1020.0],
        }
    )

    save_derived_parquet(
        frame=frame,
        kind="open_interest",
        symbol="ETH/USDT",
        timeframe="1h",
        data_dir=tmp_path,
    )
    loaded = load_derived_parquet(
        kind="open_interest",
        symbol="ETH/USDT",
        timeframe="1h",
        data_dir=tmp_path,
    )
    pd.testing.assert_frame_equal(loaded, frame)
