from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.stage34.data_snapshot import audit_and_complete_snapshot
from buffmini.utils.hashing import stable_hash


def _write_raw(base: Path, symbol: str, rows: int = 600) -> None:
    safe = symbol.replace("/", "-").replace(":", "-")
    path = Path(base) / "binance" / safe / "1m.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range("2025-01-01", periods=rows, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        }
    )
    frame.to_parquet(path, index=False)


def test_stage34_snapshot_audit_deterministic_and_aligned(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    canonical = tmp_path / "canonical"
    derived = tmp_path / "derived"
    _write_raw(raw, "BTC/USDT", rows=600)
    _write_raw(raw, "ETH/USDT", rows=540)

    first = audit_and_complete_snapshot(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframes=["1m", "5m"],
        exchange="binance",
        raw_dir=raw,
        canonical_dir=canonical,
        derived_dir=derived,
    )
    second = audit_and_complete_snapshot(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframes=["1m", "5m"],
        exchange="binance",
        raw_dir=raw,
        canonical_dir=canonical,
        derived_dir=derived,
    )
    assert first["resolved_end_ts"] is not None
    assert stable_hash(first, length=16) == stable_hash(second, length=16)
    assert len(first["rows"]) == 4

    five_rows = [row for row in first["rows"] if row["timeframe"] == "5m"]
    assert len(five_rows) == 2
    for row in five_rows:
        frame = pd.read_parquet(row["path"])
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        assert ((ts.dt.minute % 5) == 0).all()

