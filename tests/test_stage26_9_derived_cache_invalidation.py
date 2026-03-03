from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from buffmini.data.derived_tf import get_timeframe


def _canonical_file(root: Path, symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "-").replace(":", "-")
    return root / "binance" / safe / f"{timeframe}.parquet"


def _write_canonical(root: Path, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
    path = _canonical_file(root, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_stage26_9_derived_cache_invalidation(tmp_path: Path) -> None:
    canonical_dir = tmp_path / "canonical"
    derived_dir = tmp_path / "derived"
    symbol = "BTC/USDT"

    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01T00:00:00Z", periods=12, freq="1h", tz="UTC"),
            "open": [100.0 + i for i in range(12)],
            "high": [100.5 + i for i in range(12)],
            "low": [99.5 + i for i in range(12)],
            "close": [100.2 + i for i in range(12)],
            "volume": [10.0 for _ in range(12)],
        }
    )
    _write_canonical(canonical_dir, symbol, "1h", base)

    first = get_timeframe(symbol=symbol, timeframe="3h", canonical_dir=canonical_dir, derived_dir=derived_dir)
    assert first.cache_hit is False
    assert first.cache_path.exists()

    second = get_timeframe(symbol=symbol, timeframe="3h", canonical_dir=canonical_dir, derived_dir=derived_dir)
    assert second.cache_hit is True

    mutated = base.copy()
    mutated.loc[0, "close"] = float(mutated.loc[0, "close"]) + 1.0
    _write_canonical(canonical_dir, symbol, "1h", mutated)

    third = get_timeframe(symbol=symbol, timeframe="3h", canonical_dir=canonical_dir, derived_dir=derived_dir)
    assert third.cache_hit is False

    meta = json.loads(third.meta_path.read_text(encoding="utf-8"))
    assert str(meta.get("source_timeframe")) == "1h"
    assert str(meta.get("timeframe")) == "3h"
    assert str(meta.get("target_hash")) == str(third.data_hash)
