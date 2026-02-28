from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.config import load_config
from buffmini.data.cache import FeatureFrameCache, ohlcv_data_hash
from buffmini.data.features import calculate_features
from buffmini.data.store import build_data_store
from buffmini.data.storage import save_parquet
from buffmini.utils.hashing import stable_hash


def _synthetic_1m(rows: int = 720) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="1min", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.2, size=rows))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.01, 0.2, size=rows)
    low = np.minimum(open_, close) - rng.uniform(0.01, 0.2, size=rows)
    volume = rng.uniform(50.0, 500.0, size=rows)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_resample_and_feature_cache_hit_rate(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    derived_dir = tmp_path / "derived"
    feature_cache_dir = tmp_path / "features_cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    symbol = "BTC/USDT"
    save_parquet(_synthetic_1m(900), symbol=symbol, timeframe="1m", data_dir=raw_dir)

    # First run warms derived cache.
    store_first = build_data_store(
        backend="parquet",
        data_dir=raw_dir,
        base_timeframe="1m",
        resample_source="base",
        derived_dir=derived_dir,
        partial_last_bucket=False,
    )
    frame_first = store_first.load_ohlcv(symbol=symbol, timeframe="1h")
    assert len(frame_first) > 0

    # Second run should hit derived cache.
    store_second = build_data_store(
        backend="parquet",
        data_dir=raw_dir,
        base_timeframe="1m",
        resample_source="base",
        derived_dir=derived_dir,
        partial_last_bucket=False,
    )
    frame_second = store_second.load_ohlcv(symbol=symbol, timeframe="1h")
    assert len(frame_second) == len(frame_first)
    assert store_second.derived_cache.stats.hit_rate >= 0.99

    cfg = load_config(Path("configs/default.yaml"))
    cfg["data"]["include_futures_extras"] = False
    feature_cache_first = FeatureFrameCache(root_dir=feature_cache_dir)
    data_hash = ohlcv_data_hash(frame_second)
    params_hash = stable_hash({"timeframe": "1h", "extras": False}, length=16)
    key = feature_cache_first.key(symbol=symbol, timeframe="1h", data_hash=data_hash, params_hash=params_hash)
    _, first_hit = feature_cache_first.get_or_build(
        key=key,
        builder=lambda: calculate_features(frame_second, config=cfg, symbol=symbol, timeframe="1h"),
        meta={"symbol": symbol, "timeframe": "1h", "data_hash": data_hash, "params_hash": params_hash},
    )
    assert first_hit is False

    feature_cache_second = FeatureFrameCache(root_dir=feature_cache_dir)
    _, second_hit = feature_cache_second.get_or_build(
        key=key,
        builder=lambda: calculate_features(frame_second, config=cfg, symbol=symbol, timeframe="1h"),
        meta={"symbol": symbol, "timeframe": "1h", "data_hash": data_hash, "params_hash": params_hash},
    )
    assert second_hit is True
    assert feature_cache_second.stats.hit_rate >= 0.99

