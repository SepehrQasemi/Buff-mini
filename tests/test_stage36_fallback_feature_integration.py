from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.data.derived_store import save_derived_parquet
from buffmini.data.features import calculate_features
from buffmini.data.features_futures import registered_futures_feature_columns
from buffmini.validation.leakage_harness import compare_series_no_future_leakage


def _ohlcv(rows: int = 16) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="1h", tz="UTC")
    base = 100.0 + (pd.Series(range(rows), dtype=float) * 0.2)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": base,
            "high": base + 0.4,
            "low": base - 0.4,
            "close": base + 0.1,
            "volume": 1000.0,
        }
    )


def _config_include_futures_extras() -> dict:
    return {
        "data": {
            "include_futures_extras": True,
            "futures_extras": {
                "timeframe": "1h",
                "max_fill_gap_bars": 8,
                "funding": {"z_windows": [30, 90], "trend_window": 24, "abs_pctl_window": 180, "extreme_pctl": 0.95},
                "open_interest": {"chg_windows": [1, 24], "z_window": 30, "oi_to_volume_window": 24},
            },
        }
    }


def test_stage36_fallback_alignment_no_future_leakage(tmp_path: Path) -> None:
    bars = _ohlcv(12)
    symbol = "BTC/USDT"

    funding = pd.DataFrame(
        {
            "timestamp": [bars["timestamp"].iloc[3], bars["timestamp"].iloc[10]],
            "funding_rate": [0.001, 0.002],
        }
    )
    oi = pd.DataFrame(
        {
            "timestamp": [bars["timestamp"].iloc[2], bars["timestamp"].iloc[9]],
            "open_interest": [1000.0, 1200.0],
        }
    )
    save_derived_parquet(frame=funding, kind="funding", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    save_derived_parquet(frame=oi, kind="open_interest", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    baseline = calculate_features(
        bars,
        config=_config_include_futures_extras(),
        symbol=symbol,
        timeframe="1h",
        derived_data_dir=tmp_path,
    )["funding_rate"]

    shocked = funding.copy()
    shocked.loc[shocked.index == 1, "funding_rate"] = 9.999
    save_derived_parquet(frame=shocked, kind="funding", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    after = calculate_features(
        bars,
        config=_config_include_futures_extras(),
        symbol=symbol,
        timeframe="1h",
        derived_data_dir=tmp_path,
    )["funding_rate"]

    leaked, _, _ = compare_series_no_future_leakage(baseline=baseline, shocked=after, safe_end=8, tol=1e-12)
    assert leaked is False


def test_stage36_missing_open_interest_file_is_tolerated(tmp_path: Path) -> None:
    bars = _ohlcv(20)
    symbol = "ETH/USDT"
    funding = pd.DataFrame(
        {
            "timestamp": bars["timestamp"].iloc[::4],
            "funding_rate": [0.001] * 5,
        }
    )
    save_derived_parquet(frame=funding, kind="funding", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    # Intentionally omit open_interest derived parquet.
    out = calculate_features(
        bars,
        config=_config_include_futures_extras(),
        symbol=symbol,
        timeframe="1h",
        derived_data_dir=tmp_path,
    )
    assert "oi" in out.columns
    assert out["oi"].isna().all()
    assert "funding_rate" in out.columns
    assert out["funding_rate"].notna().sum() > 0


def test_stage36_feature_schema_with_fallback_extras_enabled(tmp_path: Path) -> None:
    bars = _ohlcv(32)
    symbol = "BTC/USDT"
    funding = pd.DataFrame(
        {
            "timestamp": bars["timestamp"].iloc[::3],
            "funding_rate": [0.001 + (i * 0.0001) for i in range(11)],
        }
    )
    oi = pd.DataFrame(
        {
            "timestamp": bars["timestamp"].iloc[::3],
            "open_interest": [1000.0 + (i * 10.0) for i in range(11)],
        }
    )
    save_derived_parquet(frame=funding, kind="funding", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    save_derived_parquet(frame=oi, kind="open_interest", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    out = calculate_features(
        bars,
        config=_config_include_futures_extras(),
        symbol=symbol,
        timeframe="1h",
        derived_data_dir=tmp_path,
    )
    expected = set(registered_futures_feature_columns())
    assert expected.issubset(set(out.columns))
    assert len(out) == len(bars)
