from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.stage34.dataset_builder import DatasetConfig, build_stage34_dataset
from buffmini.stage34.data_snapshot import audit_and_complete_snapshot
from buffmini.utils.hashing import stable_hash


def _write_raw(base: Path, symbol: str, rows: int = 600) -> None:
    safe = symbol.replace("/", "-").replace(":", "-")
    path = Path(base) / "binance" / safe / "1m.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range("2025-01-01", periods=rows, freq="1min", tz="UTC")
    idx = pd.Series(range(rows), dtype=float)
    base_px = 100.0 + 0.01 * idx + 0.5 * (idx % 7)
    volume = 1000.0 + (idx % 17) * 5.0
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": base_px,
            "high": base_px + 0.8,
            "low": base_px - 0.8,
            "close": base_px + ((idx % 3) - 1.0) * 0.2,
            "volume": volume,
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


def test_stage34_dataset_builder_contract(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    canonical = tmp_path / "canonical"
    derived = tmp_path / "derived"
    _write_raw(raw, "BTC/USDT", rows=2000)
    _write_raw(raw, "ETH/USDT", rows=1800)
    audit = audit_and_complete_snapshot(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframes=["1m", "15m", "1h"],
        exchange="binance",
        raw_dir=raw,
        canonical_dir=canonical,
        derived_dir=derived,
    )
    dataset, meta = build_stage34_dataset(
        cfg=DatasetConfig(
            symbols=("BTC/USDT", "ETH/USDT"),
            timeframes=("15m", "1h"),
            max_rows_per_symbol=10_000,
            max_features=120,
            horizons_hours=(24, 72),
            resolved_end_ts=str(audit.get("resolved_end_ts")),
            exchange="binance",
        ),
        canonical_dir=canonical,
        derived_dir=derived,
    )
    assert not dataset.empty
    required = {"timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe", "label_primary", "label_auxiliary"}
    assert required.issubset(set(dataset.columns))
    assert dataset["label_primary"].isin([-1, 0, 1]).all()
    assert dataset["label_auxiliary"].notna().all()
    assert int(meta["rows_total"]) == int(dataset.shape[0])
    assert str(meta["dataset_hash"])
