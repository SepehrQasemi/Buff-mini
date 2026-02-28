"""Deterministic MTF feature cache utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from buffmini.constants import DATA_DIR
from buffmini.utils.hashing import stable_hash


DEFAULT_MTF_CACHE_DIR = DATA_DIR / "features_mtf"


def build_cache_key(
    *,
    symbol: str,
    base_data_hash: str,
    target_timeframe: str,
    feature_pack_version: str,
    params: dict[str, Any] | None,
    layer_name: str = "",
) -> str:
    """Build stable cache key for MTF feature artifacts."""

    payload = {
        "symbol": str(symbol),
        "base_data_hash": str(base_data_hash),
        "target_timeframe": str(target_timeframe),
        "feature_pack_version": str(feature_pack_version),
        "params": dict(params or {}),
        "layer_name": str(layer_name),
    }
    return stable_hash(payload, length=24)


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


class MtfFeatureCache:
    """Simple deterministic parquet cache for MTF feature frames."""

    def __init__(self, root_dir: Path = DEFAULT_MTF_CACHE_DIR) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()

    def cache_path(self, cache_key: str) -> Path:
        key = str(cache_key).strip()
        if not key:
            raise ValueError("cache_key must be non-empty")
        return self.root_dir / f"{key}.parquet"

    def meta_path(self, cache_key: str) -> Path:
        return self.root_dir / f"{cache_key}.meta.json"

    def load(self, cache_key: str) -> pd.DataFrame | None:
        path = self.cache_path(cache_key)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def store(self, cache_key: str, frame: pd.DataFrame, meta: dict[str, Any] | None = None) -> Path:
        path = self.cache_path(cache_key)
        frame.to_parquet(path, index=False)
        payload = dict(meta or {})
        payload["cache_key"] = str(cache_key)
        self.meta_path(cache_key).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        return path

    def get_or_compute(
        self,
        cache_key: str,
        compute_fn: Callable[[], pd.DataFrame],
        *,
        meta: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, bool]:
        cached = self.load(cache_key)
        if cached is not None:
            self.stats.hits += 1
            return cached, True
        self.stats.misses += 1
        computed = compute_fn()
        self.store(cache_key, computed, meta=meta)
        return computed, False

