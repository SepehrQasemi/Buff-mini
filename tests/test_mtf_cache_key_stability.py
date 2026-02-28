from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.mtf.cache import MtfFeatureCache, build_cache_key


def test_mtf_cache_key_is_stable_and_sensitive_to_inputs() -> None:
    left = build_cache_key(
        symbol="BTC/USDT",
        base_data_hash="abc123",
        target_timeframe="4h",
        feature_pack_version="v1",
        params={"ema_fast": 50, "ema_slow": 200},
        layer_name="htf_4h",
    )
    right = build_cache_key(
        symbol="BTC/USDT",
        base_data_hash="abc123",
        target_timeframe="4h",
        feature_pack_version="v1",
        params={"ema_fast": 50, "ema_slow": 200},
        layer_name="htf_4h",
    )
    changed = build_cache_key(
        symbol="BTC/USDT",
        base_data_hash="abc123",
        target_timeframe="4h",
        feature_pack_version="v1",
        params={"ema_fast": 21, "ema_slow": 200},
        layer_name="htf_4h",
    )
    assert left == right
    assert left != changed


def test_mtf_cache_hit_and_miss_flow(tmp_path: Path) -> None:
    cache = MtfFeatureCache(root_dir=tmp_path / "mtf_cache")
    key = build_cache_key(
        symbol="ETH/USDT",
        base_data_hash="hash01",
        target_timeframe="4h",
        feature_pack_version="v1",
        params={"atr_window": 14},
        layer_name="htf_4h",
    )
    calls = {"count": 0}

    def _compute() -> pd.DataFrame:
        calls["count"] += 1
        return pd.DataFrame({"x": [1, 2, 3]})

    first, hit_first = cache.get_or_compute(key, _compute, meta={"symbol": "ETH/USDT"})
    second, hit_second = cache.get_or_compute(key, _compute, meta={"symbol": "ETH/USDT"})

    assert bool(hit_first) is False
    assert bool(hit_second) is True
    assert calls["count"] == 1
    pd.testing.assert_frame_equal(first, second)
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert 0.0 <= cache.stats.hit_rate <= 1.0

