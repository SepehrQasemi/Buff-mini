from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.baselines.stage0 import stage06_strategies
from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.data.cache import FeatureFrameCache, compute_features_cached, ohlcv_data_hash
from buffmini.data.features import calculate_features
from buffmini.data.storage import save_parquet
from buffmini.data.store import build_data_store
from buffmini.validation.leakage_harness import synthetic_ohlcv
from buffmini.validation.walkforward_v2 import build_windows, evaluate_candidate_on_window


def _synthetic_1m(rows: int = 240) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=int(rows), freq="min", tz="UTC")
    base = pd.Series(range(int(rows)), dtype=float) * 0.01 + 100.0
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": base,
            "high": base + 0.05,
            "low": base - 0.05,
            "close": base,
            "volume": 100.0,
        }
    )


def test_derived_cache_avoids_recompute(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    derived_dir = tmp_path / "derived"
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    symbol = "BTC/USDT"
    save_parquet(frame=_synthetic_1m(600), symbol=symbol, timeframe="1m", data_dir=raw_dir)

    import buffmini.data.store as store_mod

    original_resample = store_mod.resample_ohlcv
    calls = {"count": 0}

    def spy_resample(*args, **kwargs):
        calls["count"] += 1
        return original_resample(*args, **kwargs)

    monkeypatch.setattr(store_mod, "resample_ohlcv", spy_resample)

    store = build_data_store(
        backend="parquet",
        data_dir=raw_dir,
        base_timeframe="1m",
        resample_source="base",
        derived_dir=derived_dir,
        config_hash="cfg_hash_test",
        resolved_end_ts="2025-01-01T09:59:00+00:00",
    )

    first = store.load_ohlcv(symbol=symbol, timeframe="1h")
    second = store.load_ohlcv(symbol=symbol, timeframe="1h")

    assert len(first) > 0
    assert first.equals(second)
    assert calls["count"] == 1
    assert int(store.derived_cache.stats.misses) == 1
    assert int(store.derived_cache.stats.hits) == 1


def test_walkforward_features_computed_once_per_tf(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    raw = synthetic_ohlcv(rows=24 * 500, seed=11)
    feature_cache = FeatureFrameCache(root_dir=tmp_path / "feature_cache")
    strategy = stage06_strategies()[0]
    candidate = {"strategy": strategy, "symbol": "BTC/USDT", "gating_mode": "none"}

    dataset_key_hash = compute_config_hash(config)
    data_hash = ohlcv_data_hash(raw)
    calls = {"count": 0}

    def builder():
        calls["count"] += 1
        return calculate_features(raw, config=config, symbol="BTC/USDT", timeframe="1h")

    feature_frame, _, _ = compute_features_cached(
        cache=feature_cache,
        symbol="BTC/USDT",
        timeframe="1h",
        resolved_end_ts=str(config.get("universe", {}).get("resolved_end_ts") or ""),
        feature_config_hash=dataset_key_hash,
        data_hash=data_hash,
        builder=builder,
    )

    start_ts = feature_frame["timestamp"].iloc[0]
    end_ts = feature_frame["timestamp"].iloc[-1] + feature_frame["timestamp"].diff().dropna().median()
    windows = build_windows(
        start_ts=start_ts,
        end_ts=end_ts,
        train_days=180,
        holdout_days=30,
        forward_days=30,
        step_days=30,
        reserve_tail_days=0,
    )
    assert len(windows) >= 2

    for window in windows[:3]:
        # Simulate walk-forward loop repeatedly requesting feature data for the same dataset key.
        reused, from_cache, _ = compute_features_cached(
            cache=feature_cache,
            symbol="BTC/USDT",
            timeframe="1h",
            resolved_end_ts=str(config.get("universe", {}).get("resolved_end_ts") or ""),
            feature_config_hash=dataset_key_hash,
            data_hash=data_hash,
            builder=builder,
        )
        assert bool(from_cache)
        metrics = evaluate_candidate_on_window(candidate=candidate, data=reused, window_triplet=window, cfg=config)
        assert "forward_expectancy" in metrics

    assert calls["count"] == 1
    assert int(feature_cache.stats.compute_calls_per_tf.get("1h", 0)) == 1
