from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from buffmini.data.canonical_raw import (
    detect_gaps_minutes,
    file_sha256,
    prepare_frame,
    raw_meta_path,
    raw_path,
)


def test_stage26_9_raw_prepare_sorts_and_dedups() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01T00:02:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:01:00Z",
                "2025-01-01T00:01:00Z",
            ],
            "open": [3.0, 1.0, 2.0, 2.1],
            "high": [3.1, 1.1, 2.1, 2.2],
            "low": [2.9, 0.9, 1.9, 2.0],
            "close": [3.0, 1.0, 2.0, 2.1],
            "volume": [1.0, 1.0, 1.0, 1.5],
        }
    )
    out = prepare_frame(frame)
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert out["timestamp"].is_monotonic_increasing
    assert int(out["timestamp"].duplicated().sum()) == 0
    assert int(out.shape[0]) == 3


def test_stage26_9_gap_detection() -> None:
    ts = pd.to_datetime(
        [
            "2025-01-01T00:00:00Z",
            "2025-01-01T00:01:00Z",
            "2025-01-01T00:05:00Z",
            "2025-01-01T00:06:00Z",
        ],
        utc=True,
    )
    gaps = detect_gaps_minutes(pd.Series(ts), expected_minutes=1)
    assert gaps.gaps_detected == 1
    assert gaps.max_gap_minutes == 4


def test_stage26_9_meta_schema_and_sha(tmp_path: Path) -> None:
    symbol = "BTC/USDT"
    exchange = "binance"
    timeframe = "1m"
    path = raw_path(data_dir=tmp_path, exchange=exchange, symbol=symbol, timeframe=timeframe)
    meta_path = raw_meta_path(data_dir=tmp_path, exchange=exchange, symbol=symbol, timeframe=timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01T00:00:00Z", periods=5, freq="1min", tz="UTC"),
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0, 1, 2, 3, 4],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [10, 10, 10, 10, 10],
        }
    )
    frame.to_parquet(path, index=False)
    payload = {
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "requested_years": 4,
        "actual_start_ts": pd.Timestamp(frame["timestamp"].iloc[0]).isoformat(),
        "actual_end_ts": pd.Timestamp(frame["timestamp"].iloc[-1]).isoformat(),
        "candle_count": int(frame.shape[0]),
        "sha256": file_sha256(path),
        "gaps_detected": {"count": 0, "max_gap_minutes": 0},
        "generator_version": "stage26_9",
    }
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in (
        "exchange",
        "symbol",
        "timeframe",
        "requested_years",
        "actual_start_ts",
        "actual_end_ts",
        "candle_count",
        "sha256",
        "gaps_detected",
        "generator_version",
    ):
        assert key in loaded
    assert str(loaded["sha256"]) == file_sha256(path)
