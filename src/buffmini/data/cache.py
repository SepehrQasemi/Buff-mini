"""Data-level cache helpers for derived timeframe bars and feature frames."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
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
    build_seconds: float = 0.0

    @property
    def total(self) -> int:
        return int(self.hits + self.misses)

    @property
    def hit_rate(self) -> float:
        total = self.total
        if total <= 0:
            return 0.0
        return float(self.hits / total)


@dataclass
class FeatureCacheStats(CacheStats):
    compute_calls_per_tf: dict[str, int] = field(default_factory=dict)


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
        legacy_key = stable_hash(
            {
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "source_hash": str(source_hash),
                "settings_hash": str(settings_hash),
            },
            length=24,
        )
        return self.get_or_build_by_key(
            symbol=symbol,
            timeframe=timeframe,
            cache_key=legacy_key,
            builder=builder,
            meta={
                "source_hash": str(source_hash),
                "settings_hash": str(settings_hash),
            },
        )

    def get_or_build_by_key(
        self,
        *,
        symbol: str,
        timeframe: str,
        cache_key: str,
        builder: Callable[[], pd.DataFrame],
        meta: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, bool]:
        cached = self._load_if_key(
            symbol=symbol,
            timeframe=timeframe,
            cache_key=cache_key,
        )
        if cached is not None:
            self.stats.hits += 1
            return cached, True
        self.stats.misses += 1
        started = time.perf_counter()
        built = builder()
        self.stats.build_seconds += float(time.perf_counter() - started)
        self.store_by_key(
            symbol=symbol,
            timeframe=timeframe,
            frame=built,
            cache_key=cache_key,
            meta=meta or {},
        )
        return built, False

    def store_by_key(
        self,
        *,
        symbol: str,
        timeframe: str,
        frame: pd.DataFrame,
        cache_key: str,
        meta: dict[str, Any],
    ) -> None:
        path = self.cache_path(symbol=symbol, timeframe=timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        payload = dict(meta)
        payload.update(
            {
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "cache_key": str(cache_key),
                "rows": int(len(frame)),
                "frame_hash": ohlcv_data_hash(frame),
            }
        )
        self.meta_path(symbol=symbol, timeframe=timeframe).write_text(
            json.dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    def store(
        self,
        *,
        symbol: str,
        timeframe: str,
        frame: pd.DataFrame,
        source_hash: str,
        settings_hash: str,
    ) -> None:
        legacy_key = stable_hash(
            {
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "source_hash": str(source_hash),
                "settings_hash": str(settings_hash),
            },
            length=24,
        )
        self.store_by_key(
            symbol=symbol,
            timeframe=timeframe,
            frame=frame,
            cache_key=legacy_key,
            meta={
                "source_hash": str(source_hash),
                "settings_hash": str(settings_hash),
            },
        )

    def _load_if_key(
        self,
        *,
        symbol: str,
        timeframe: str,
        cache_key: str,
    ) -> pd.DataFrame | None:
        path = self.cache_path(symbol=symbol, timeframe=timeframe)
        meta_path = self.meta_path(symbol=symbol, timeframe=timeframe)
        if not path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if str(meta.get("cache_key", "")) != str(cache_key):
            return None
        return pd.read_parquet(path)


class FeatureFrameCache:
    """Feature dataframe cache keyed by symbol/timeframe/data+params hashes."""

    def __init__(self, root_dir: Path = FEATURE_CACHE_DIR) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.stats = FeatureCacheStats()

    def key(
        self,
        *,
        symbol: str,
        timeframe: str,
        data_hash: str,
        params_hash: str,
        resolved_end_ts: str | None = None,
    ) -> str:
        return stable_hash(
            {
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "data_hash": str(data_hash),
                "params_hash": str(params_hash),
                "resolved_end_ts": str(resolved_end_ts or ""),
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
        started = time.perf_counter()
        frame = builder()
        self.stats.build_seconds += float(time.perf_counter() - started)
        timeframe = str(meta.get("timeframe", "unknown"))
        self.stats.compute_calls_per_tf[timeframe] = int(self.stats.compute_calls_per_tf.get(timeframe, 0) + 1)
        frame.to_parquet(path, index=False)
        payload = dict(meta)
        payload["key"] = str(key)
        self.meta_path(key).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        return frame, False


def derived_cache_key(
    *,
    symbol: str,
    base_tf: str,
    target_tf: str,
    resolved_end_ts: str | None,
    data_hash: str,
    config_hash: str,
) -> str:
    """Stable key for derived timeframe cache enforcement."""

    return stable_hash(
        {
            "symbol": str(symbol),
            "base_tf": str(base_tf),
            "target_tf": str(target_tf),
            "resolved_end_ts": str(resolved_end_ts or ""),
            "data_hash": str(data_hash),
            "config_hash": str(config_hash),
        },
        length=24,
    )


def get_or_build_derived_ohlcv(
    *,
    cache: DerivedTimeframeCache,
    symbol: str,
    base_tf: str,
    target_tf: str,
    start_ts: str | None,
    end_ts: str | None,
    resolved_end_ts: str | None,
    data_hash: str,
    config_hash: str,
    builder: Callable[[], pd.DataFrame],
) -> tuple[pd.DataFrame, bool]:
    """Strict derived OHLCV cache contract used by the engine."""

    cache_key = derived_cache_key(
        symbol=symbol,
        base_tf=base_tf,
        target_tf=target_tf,
        resolved_end_ts=resolved_end_ts,
        data_hash=data_hash,
        config_hash=config_hash,
    )
    return cache.get_or_build_by_key(
        symbol=symbol,
        timeframe=target_tf,
        cache_key=cache_key,
        builder=builder,
        meta={
            "base_tf": str(base_tf),
            "target_tf": str(target_tf),
            "resolved_end_ts": str(resolved_end_ts or ""),
            "data_hash": str(data_hash),
            "config_hash": str(config_hash),
            "start_ts": str(start_ts or ""),
            "end_ts": str(end_ts or ""),
        },
    )


def feature_dataset_key(
    *,
    symbol: str,
    timeframe: str,
    resolved_end_ts: str | None,
    feature_config_hash: str,
    data_hash: str,
) -> str:
    """Stable dataset key for compute-once feature caching."""

    return stable_hash(
        {
            "symbol": str(symbol),
            "timeframe": str(timeframe),
            "resolved_end_ts": str(resolved_end_ts or ""),
            "feature_config_hash": str(feature_config_hash),
            "data_hash": str(data_hash),
        },
        length=24,
    )


def compute_features_cached(
    *,
    cache: FeatureFrameCache | None,
    symbol: str,
    timeframe: str,
    resolved_end_ts: str | None,
    feature_config_hash: str,
    data_hash: str,
    builder: Callable[[], pd.DataFrame],
) -> tuple[pd.DataFrame, bool, str]:
    """Compute features once per dataset key and reuse deterministically."""

    dataset_key = feature_dataset_key(
        symbol=symbol,
        timeframe=timeframe,
        resolved_end_ts=resolved_end_ts,
        feature_config_hash=feature_config_hash,
        data_hash=data_hash,
    )
    if cache is None:
        frame = builder()
        return frame, False, dataset_key
    key = cache.key(
        symbol=symbol,
        timeframe=timeframe,
        data_hash=data_hash,
        params_hash=feature_config_hash,
        resolved_end_ts=resolved_end_ts,
    )
    frame, from_cache = cache.get_or_build(
        key=key,
        builder=builder,
        meta={
            "symbol": str(symbol),
            "timeframe": str(timeframe),
            "resolved_end_ts": str(resolved_end_ts or ""),
            "feature_config_hash": str(feature_config_hash),
            "data_hash": str(data_hash),
            "dataset_key": dataset_key,
        },
    )
    return frame, from_cache, dataset_key
