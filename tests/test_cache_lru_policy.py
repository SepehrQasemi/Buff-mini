from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from buffmini.data.cache import CacheLimits, FeatureFrameCache


def _frame(rows: int = 50) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        }
    )


def test_feature_cache_lru_max_entries_per_tf(tmp_path: Path) -> None:
    cache = FeatureFrameCache(root_dir=tmp_path / "features_cache", limits=CacheLimits(max_entries_per_tf=2, max_total_mb=2048))
    frame = _frame()

    for idx in range(4):
        key = f"k{idx}"
        out, from_cache = cache.get_or_build(
            key=key,
            builder=lambda f=frame: f,
            meta={
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "resolved_end_ts": "2025-01-01T00:00:00+00:00",
            },
        )
        assert not bool(from_cache)
        assert len(out) == len(frame)

    registry = json.loads((tmp_path / "features_cache" / "_cache_registry.json").read_text(encoding="utf-8"))
    entries = registry.get("entries", {})
    assert len(entries) == 2
    for entry in entries.values():
        path = Path(str(entry["path"]))
        assert path.exists()

    # Oldest keys should be evicted by LRU policy.
    assert not (tmp_path / "features_cache" / "k0.parquet").exists()
    assert not (tmp_path / "features_cache" / "k1.parquet").exists()
    assert (tmp_path / "features_cache" / "k2.parquet").exists()
    assert (tmp_path / "features_cache" / "k3.parquet").exists()
