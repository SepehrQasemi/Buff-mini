from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.data.features import calculate_features
from buffmini.features.extras_align import align_coinapi_extras_to_bars
from buffmini.validation.leakage_harness import compare_series_no_future_leakage


def _ohlcv(rows: int = 24) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="1h", tz="UTC")
    base = 100.0 + (pd.Series(range(rows), dtype=float) * 0.1)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.1,
            "volume": 1000.0,
        }
    )


def test_coinapi_alignment_is_deterministic_and_backward_only() -> None:
    bars = _ohlcv(8)
    funding = pd.DataFrame(
        {
            "ts": [bars["timestamp"].iloc[4], bars["timestamp"].iloc[6]],
            "funding_rate": [0.001, 0.002],
        }
    )
    aligned_a = align_coinapi_extras_to_bars(bars, funding=funding)
    aligned_b = align_coinapi_extras_to_bars(bars, funding=funding)
    pd.testing.assert_frame_equal(aligned_a, aligned_b)

    # Future event must not leak backward.
    assert pd.isna(aligned_a.loc[0, "funding_rate"])
    assert pd.isna(aligned_a.loc[3, "funding_rate"])
    assert float(aligned_a.loc[4, "funding_rate"]) == 0.001


def test_calculate_features_with_coinapi_extras(tmp_path: Path) -> None:
    bars = _ohlcv(16)
    symbol_dir = tmp_path / "BTC_USDT"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ts": bars["timestamp"].iloc[::4], "funding_rate": [0.001, 0.002, 0.003, 0.004]}).to_parquet(
        symbol_dir / "funding_rates.parquet",
        index=False,
    )
    pd.DataFrame({"ts": bars["timestamp"].iloc[::4], "open_interest": [1000, 1100, 1200, 1300]}).to_parquet(
        symbol_dir / "open_interest.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "ts": bars["timestamp"].iloc[::2],
            "liq_buy": [1.0] * 8,
            "liq_sell": [0.5] * 8,
            "liq_notional": [10000.0] * 8,
        }
    ).to_parquet(symbol_dir / "liquidations.parquet", index=False)

    cfg = {
        "data": {"include_futures_extras": False},
        "features": {
            "extras": {
                "enabled": True,
                "sources": ["coinapi"],
                "max_staleness": {"funding_rates": "24h", "open_interest": "24h", "liquidations": "24h"},
            }
        },
    }
    out = calculate_features(
        bars,
        config=cfg,
        symbol="BTC/USDT",
        timeframe="1h",
        coinapi_canonical_dir=tmp_path,
    )
    assert "funding_rate" in out.columns
    assert "oi" in out.columns
    assert "liq_buy" in out.columns
    assert out["funding_rate"].notna().sum() > 0


def test_synthetic_future_spike_in_extras_is_caught_by_leak_probe() -> None:
    bars = _ohlcv(10)
    base_extras = pd.DataFrame(
        {
            "ts": [bars["timestamp"].iloc[2], bars["timestamp"].iloc[5], bars["timestamp"].iloc[8]],
            "funding_rate": [0.001, 0.002, 0.003],
        }
    )
    shocked_extras = base_extras.copy()
    shocked_extras.loc[shocked_extras.index == 2, "funding_rate"] = 9.999

    # Intentionally leaky alignment to prove the probe catches it.
    baseline = pd.merge_asof(
        bars[["timestamp"]].sort_values("timestamp"),
        base_extras.sort_values("ts"),
        left_on="timestamp",
        right_on="ts",
        direction="forward",
    )["funding_rate"]
    shocked = pd.merge_asof(
        bars[["timestamp"]].sort_values("timestamp"),
        shocked_extras.sort_values("ts"),
        left_on="timestamp",
        right_on="ts",
        direction="forward",
    )["funding_rate"]

    leaked, _, _ = compare_series_no_future_leakage(baseline=baseline, shocked=shocked, safe_end=6, tol=1e-12)
    assert leaked is True
