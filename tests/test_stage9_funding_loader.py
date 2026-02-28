"""Stage-9.1 funding loader and derived storage tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.data.derived_store import load_derived_parquet, save_derived_parquet, write_meta_json
from buffmini.data.futures_extras import align_funding_to_ohlcv, funding_quality_report


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


def test_funding_alignment_uses_latest_event_le_candle_close() -> None:
    ohlcv = _ohlcv()
    funding = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01T00:30:00Z",
                    "2025-01-01T02:00:00Z",
                    "2025-01-01T04:20:00Z",
                ],
                utc=True,
            ),
            "funding_rate": [0.001, 0.002, -0.001],
        }
    )

    aligned = align_funding_to_ohlcv(ohlcv=ohlcv, funding=funding, timeframe="1h")

    # For candle open at 00:00 close=01:00 -> last event <=01:00 is 00:30
    assert float(aligned.loc[0, "funding_rate"]) == 0.001
    # Candle open 01:00 close=02:00 includes event at 02:00
    assert float(aligned.loc[1, "funding_rate"]) == 0.002
    # Candle open 03:00 close=04:00 still 02:00 event
    assert float(aligned.loc[3, "funding_rate"]) == 0.002
    # Candle open 04:00 close=05:00 picks 04:20 event
    assert float(aligned.loc[4, "funding_rate"]) == -0.001


def test_funding_quality_report_flags_duplicates_and_is_finite() -> None:
    frame = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T08:00:00Z",
                    "2025-01-01T08:00:00Z",
                    "2025-01-01T16:00:00Z",
                ],
                utc=True,
            ),
            "funding_rate": [0.001, 0.002, 0.003, 0.004],
        }
    )
    quality = funding_quality_report(frame)
    assert quality["rows"] == 3
    assert quality["monotonic_ts"] is True
    assert quality["duplicates"] == 1
    assert quality["finite"] is True


def test_derived_store_roundtrip_for_funding(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
            "funding_rate": [0.001, 0.002, 0.003],
        }
    )
    save_derived_parquet(frame=frame, kind="funding", symbol="BTC/USDT", timeframe="1h", data_dir=tmp_path)
    loaded = load_derived_parquet(kind="funding", symbol="BTC/USDT", timeframe="1h", data_dir=tmp_path)
    pd.testing.assert_frame_equal(loaded, frame)

    meta_path = write_meta_json(
        kind="funding",
        symbol="BTC/USDT",
        timeframe="1h",
        payload={"row_count": 3},
        data_dir=tmp_path,
    )
    assert meta_path.exists()
