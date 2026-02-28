"""Data-level cache helpers for derived timeframe bars and feature frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from buffmini.constants import DATA_DIR, DERIVED_DATA_DIR
from buffmini.data.storage import parquet_path
from buffmini.utils.hashing import stable_hash


FEATURE_CACHE_DIR = DATA_DIR / "features_cache"


def ohlcv_data_hash(frame: pd.DataFrame) -> str:
    """Stable hash for OHLCV content based on row-wise hashes."""

    if frame.empty:
        return stable_hash({"rows": 0}, length=16)
    work = frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).reset_index(drop=True)
    row_hash = pd.util.hash_pandas_object(work, index=False, categorize=False).astype("uint64")
    payload = {
        "rows": int(len(work)),
        "first_ts": work["timestamp"].iloc[0].isoformat() if len(work) else None,
        "last_ts": work["timestamp"].iloc[-1].isoformat() if len(work) else None,
        "row_hash_sum": int(row_hash.sum()),
        "row_hash_xor": int(row_hash.values[0] if len(row_hash) == 1 else row_hash.values[0] ^ row_hash.values[-1]),
    }
    return stable_hash(payload, length=16)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return int(self.hits + self.misses)

    @property
    def hit_rate(self) -> float:
        total = self.total
        if total <= 0:
            return 0.0
        return float(self.hits / total)


class DerivedTimeframeCache:
    """Cache for derived timeframe bars under data/derived."""

    def __init__(self, derived_dir: Path = DERIVED_DATA_DIR) -> None:
        self.derived_dir = Path(derived_dir)
        self.derived_dir.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()

    def cache_path(self, symbol: str, timeframe: str) -> Path:
        return parquet_path(symbol=symbol, timeframe=timeframe, data_dir=self.derived_dir)

    def meta_path(self, symbol: str, timeframe: str) -> Path:
        return self.cache_path(symbol=symbol, timeframe=timeframe).with_suffix(".meta.json")

    def get_or_build(
        self,
        *,
        symbol: str,
        timeframe: str,
        source_hash: str,
        settings_hash: str,
        builder: Callable[[], pd.DataFrame],
    ) -> tuple[pd.DataFrame, bool]:
        cached = self._load_if_match(
            symbol=symbol,
            timeframe=timeframe,
            source_hash=source_hash,
            settings_hash=settings_hash,
        )
        if cached is not None:
            self.stats.hits += 1
            return cached, True
        self.stats.misses += 1
        built = builder()
        self.store(
            symbol=symbol,
            timeframe=timeframe,
            frame=built,
            source_hash=source_hash,
            settings_hash=settings_hash,
        )
        return built, False

    def store(
        self,
        *,
        symbol: str,
        timeframe: str,
        frame: pd.DataFrame,
        source_hash: str,
        settings_hash: str,
    ) -> None:
        path = self.cache_path(symbol=symbol, timeframe=timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        meta = {
            "symbol": str(symbol),
            "timeframe": str(timeframe),
            "source_hash": str(source_hash),
            "settings_hash": str(settings_hash),
            "rows": int(len(frame)),
            "frame_hash": ohlcv_data_hash(frame),
        }
        self.meta_path(symbol=symbol, timeframe=timeframe).write_text(
            json.dumps(meta, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    def _load_if_match(
        self,
        *,
        symbol: str,
        timeframe: str,
        source_hash: str,
        settings_hash: str,
    ) -> pd.DataFrame | None:
        path = self.cache_path(symbol=symbol, timeframe=timeframe)
        meta_path = self.meta_path(symbol=symbol, timeframe=timeframe)
        if not path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if str(meta.get("source_hash")) != str(source_hash):
            return None
        if str(meta.get("settings_hash")) != str(settings_hash):
            return None
        return pd.read_parquet(path)


class FeatureFrameCache:
    """Feature dataframe cache keyed by symbol/timeframe/data+params hashes."""

    def __init__(self, root_dir: Path = FEATURE_CACHE_DIR) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()

    def key(self, *, symbol: str, timeframe: str, data_hash: str, params_hash: str) -> str:
        return stable_hash(
            {
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "data_hash": str(data_hash),
                "params_hash": str(params_hash),
            },
            length=24,
        )

    def cache_path(self, key: str) -> Path:
        return self.root_dir / f"{key}.parquet"

    def meta_path(self, key: str) -> Path:
        return self.root_dir / f"{key}.meta.json"

    def get_or_build(self, *, key: str, builder: Callable[[], pd.DataFrame], meta: dict[str, Any]) -> tuple[pd.DataFrame, bool]:
        path = self.cache_path(key)
        if path.exists():
            self.stats.hits += 1
            return pd.read_parquet(path), True
        self.stats.misses += 1
        frame = builder()
        frame.to_parquet(path, index=False)
        payload = dict(meta)
        payload["key"] = str(key)
        self.meta_path(key).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        return frame, False

