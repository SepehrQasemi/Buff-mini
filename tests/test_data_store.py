"""Data store tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from buffmini.data.storage import load_parquet
from buffmini.data.store import ParquetStore, build_data_store


def _sample_ohlcv() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01T00:00:00Z", periods=5, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )


def test_parquet_store_roundtrip_and_coverage(tmp_path: Path) -> None:
    store = ParquetStore(data_dir=tmp_path / "raw")
    frame = _sample_ohlcv()

    store.save_ohlcv(symbol="BTC/USDT", timeframe="1h", df=frame)
    loaded = store.load_ohlcv(symbol="BTC/USDT", timeframe="1h")

    assert_frame_equal(loaded.reset_index(drop=True), frame.reset_index(drop=True), check_dtype=False)

    coverage = store.coverage(symbol="BTC/USDT", timeframe="1h")
    assert coverage["exists"] is True
    assert coverage["rows"] == len(frame)
    assert coverage["start"] == pd.Timestamp(frame["timestamp"].iloc[0]).tz_convert("UTC").isoformat()
    assert coverage["end"] == pd.Timestamp(frame["timestamp"].iloc[-1]).tz_convert("UTC").isoformat()


def test_build_data_store_parquet_matches_storage_load(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    frame = _sample_ohlcv()
    store = build_data_store(backend="parquet", data_dir=raw_dir)
    store.save_ohlcv(symbol="ETH/USDT", timeframe="1h", df=frame)

    loaded_store = store.load_ohlcv(symbol="ETH/USDT", timeframe="1h")
    loaded_direct = load_parquet(symbol="ETH/USDT", timeframe="1h", data_dir=raw_dir)

    assert_frame_equal(loaded_store.reset_index(drop=True), loaded_direct.reset_index(drop=True), check_dtype=False)
