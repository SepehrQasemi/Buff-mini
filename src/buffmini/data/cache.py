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


@dataclass(frozen=True)
class CacheLimits:
    max_entries_per_tf: int = 5
    max_total_mb: float = 2048.0


class DerivedTimeframeCache:
    """Cache for derived timeframe bars under data/derived."""

    def __init__(self, derived_dir: Path = DERIVED_DATA_DIR, *, limits: CacheLimits | None = None) -> None:
        self.derived_dir = Path(derived_dir)
        self.derived_dir.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()
        self.limits = limits or CacheLimits()
        self.registry_path = self.derived_dir / "_cache_registry.json"

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
            self._touch_registry_entry(key=cache_key)
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
        self._register_cache_entry(
            key=str(cache_key),
            symbol=str(symbol),
            timeframe=str(timeframe),
            resolved_end_ts=str(payload.get("resolved_end_ts", "")),
            path=path,
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

    def _load_registry(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"entries": {}}
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception:
            return {"entries": {}}
        entries = payload.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}
        return {"entries": entries}

    def _write_registry(self, payload: dict[str, Any]) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    def _register_cache_entry(self, *, key: str, symbol: str, timeframe: str, resolved_end_ts: str, path: Path) -> None:
        payload = self._load_registry()
        entries = payload["entries"]
        now = _utc_now_iso()
        size_bytes = int(path.stat().st_size) if path.exists() else 0
        entries[str(key)] = {
            "key": str(key),
            "timestamp": now,
            "last_access_at": now,
            "symbol": str(symbol),
            "timeframe": str(timeframe),
            "resolved_end_ts": str(resolved_end_ts),
            "size_bytes": size_bytes,
            "path": str(path),
        }
        payload["entries"] = entries
        payload = _sanitize_registry_entries(payload)
        payload = _enforce_registry_limits(payload, limits=self.limits)
        self._apply_registry_deletions(payload)
        self._write_registry(payload)

    def _touch_registry_entry(self, *, key: str) -> None:
        payload = self._load_registry()
        entries = payload["entries"]
        if str(key) in entries:
            entries[str(key)]["last_access_at"] = _utc_now_iso()
            payload["entries"] = entries
            self._write_registry(payload)

    def _apply_registry_deletions(self, payload: dict[str, Any]) -> None:
        for entry in payload.get("_delete", []):
            path = Path(str(entry.get("path", "")))
            if path.exists():
                path.unlink(missing_ok=True)
            meta = path.with_suffix(".meta.json")
            if meta.exists():
                meta.unlink(missing_ok=True)


class FeatureFrameCache:
    """Feature dataframe cache keyed by symbol/timeframe/data+params hashes."""

    def __init__(self, root_dir: Path = FEATURE_CACHE_DIR, *, limits: CacheLimits | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.stats = FeatureCacheStats()
        self.limits = limits or CacheLimits()
        self.registry_path = self.root_dir / "_cache_registry.json"

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
            self._touch_registry_entry(key=key)
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
        self._register_cache_entry(
            key=str(key),
            symbol=str(meta.get("symbol", "")),
            timeframe=timeframe,
            resolved_end_ts=str(meta.get("resolved_end_ts", "")),
            path=path,
        )
        return frame, False

    def _load_registry(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"entries": {}}
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception:
            return {"entries": {}}
        entries = payload.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}
        return {"entries": entries}

    def _write_registry(self, payload: dict[str, Any]) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    def _register_cache_entry(self, *, key: str, symbol: str, timeframe: str, resolved_end_ts: str, path: Path) -> None:
        payload = self._load_registry()
        entries = payload["entries"]
        now = _utc_now_iso()
        size_bytes = int(path.stat().st_size) if path.exists() else 0
        entries[str(key)] = {
            "key": str(key),
            "timestamp": now,
            "last_access_at": now,
            "symbol": str(symbol),
            "timeframe": str(timeframe),
            "resolved_end_ts": str(resolved_end_ts),
            "size_bytes": size_bytes,
            "path": str(path),
        }
        payload["entries"] = entries
        payload = _sanitize_registry_entries(payload)
        payload = _enforce_registry_limits(payload, limits=self.limits)
        self._apply_registry_deletions(payload)
        self._write_registry(payload)

    def _touch_registry_entry(self, *, key: str) -> None:
        payload = self._load_registry()
        entries = payload["entries"]
        if str(key) in entries:
            entries[str(key)]["last_access_at"] = _utc_now_iso()
            payload["entries"] = entries
            self._write_registry(payload)

    def _apply_registry_deletions(self, payload: dict[str, Any]) -> None:
        for entry in payload.get("_delete", []):
            path = Path(str(entry.get("path", "")))
            if path.exists():
                path.unlink(missing_ok=True)
            meta = path.with_suffix(".meta.json")
            if meta.exists():
                meta.unlink(missing_ok=True)


class FeatureComputeSession:
    """In-process dedup boundary: compute each dataset_key once per run."""

    def __init__(self, cache: FeatureFrameCache | None) -> None:
        self.cache = cache
        self._frames: dict[str, pd.DataFrame] = {}
        self.requests_per_tf: dict[str, int] = {}
        self.memory_hits: int = 0

    def get_or_build(
        self,
        *,
        symbol: str,
        timeframe: str,
        resolved_end_ts: str | None,
        feature_config_hash: str,
        data_hash: str,
        builder: Callable[[], pd.DataFrame],
    ) -> tuple[pd.DataFrame, bool, str]:
        tf = str(timeframe)
        self.requests_per_tf[tf] = int(self.requests_per_tf.get(tf, 0) + 1)
        dataset_key = feature_dataset_key(
            symbol=symbol,
            timeframe=timeframe,
            resolved_end_ts=resolved_end_ts,
            feature_config_hash=feature_config_hash,
            data_hash=data_hash,
        )
        if dataset_key in self._frames:
            self.memory_hits += 1
            return self._frames[dataset_key], True, dataset_key
        frame, from_cache, dataset_key_out = compute_features_cached(
            cache=self.cache,
            symbol=symbol,
            timeframe=timeframe,
            resolved_end_ts=resolved_end_ts,
            feature_config_hash=feature_config_hash,
            data_hash=data_hash,
            builder=builder,
        )
        self._frames[dataset_key] = frame
        return frame, from_cache, dataset_key_out


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


def _sanitize_registry_entries(payload: dict[str, Any]) -> dict[str, Any]:
    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        return {"entries": {}}
    clean: dict[str, dict[str, Any]] = {}
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        path = Path(str(entry.get("path", "")))
        if not path.exists():
            continue
        item = dict(entry)
        item["size_bytes"] = int(path.stat().st_size)
        item["path"] = str(path)
        clean[str(key)] = item
    return {"entries": clean}


def _enforce_registry_limits(payload: dict[str, Any], *, limits: CacheLimits) -> dict[str, Any]:
    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        return {"entries": {}}
    max_entries_per_tf = max(1, int(limits.max_entries_per_tf))
    max_total_bytes = int(float(limits.max_total_mb) * 1024.0 * 1024.0)
    to_delete: list[dict[str, Any]] = []
    working = dict(entries)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for key, entry in working.items():
        tf = str(entry.get("timeframe", "unknown"))
        bucket = dict(entry)
        bucket["_key"] = str(key)
        grouped.setdefault(tf, []).append(bucket)
    for tf, rows in grouped.items():
        if len(rows) <= max_entries_per_tf:
            continue
        rows_sorted = sorted(
            rows,
            key=lambda item: (
                str(item.get("last_access_at", "")),
                str(item.get("timestamp", "")),
                str(item.get("_key", "")),
            ),
        )
        remove_n = len(rows_sorted) - max_entries_per_tf
        for item in rows_sorted[:remove_n]:
            key = str(item.get("_key", ""))
            if key in working:
                to_delete.append(dict(item))
                del working[key]

    def total_size(rows: dict[str, Any]) -> int:
        return int(sum(int(item.get("size_bytes", 0)) for item in rows.values()))

    while total_size(working) > max_total_bytes and working:
        oldest = sorted(
            (dict(v, _key=str(k)) for k, v in working.items()),
            key=lambda item: (
                str(item.get("last_access_at", "")),
                str(item.get("timestamp", "")),
                str(item.get("_key", "")),
            ),
        )[0]
        key = str(oldest.get("_key", ""))
        to_delete.append(oldest)
        del working[key]

    out = {"entries": working}
    if to_delete:
        out["_delete"] = to_delete
    return out


def _utc_now_iso() -> str:
    ts = pd.Timestamp.utcnow()
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def cache_limits_from_config(config: dict[str, Any] | None) -> CacheLimits:
    """Resolve cache limits from config, falling back to safe defaults."""

    if not isinstance(config, dict):
        return CacheLimits()
    cache_cfg = (
        config.get("evaluation", {})
        .get("stage11_55", {})
        .get("cache", {})
    )
    if not isinstance(cache_cfg, dict):
        return CacheLimits()
    max_entries = int(cache_cfg.get("max_entries_per_tf", 5))
    max_total_mb = float(cache_cfg.get("max_total_mb", 2048))
    if max_entries < 1:
        max_entries = 1
    if max_total_mb <= 0:
        max_total_mb = 2048.0
    return CacheLimits(max_entries_per_tf=max_entries, max_total_mb=max_total_mb)
