from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.data.derived_store import save_derived_parquet
from buffmini.data.features import calculate_features
from buffmini.stage9.oi_overlay import OI_DEPENDENT_COLUMNS


def _ohlcv(freq: str, rows: int) -> pd.DataFrame:
    ts = pd.date_range("2025-07-01T00:00:00Z", periods=rows, freq=freq, tz="UTC")
    base = 100.0 + (pd.Series(range(rows), dtype=float) * 0.1)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": base,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base + 0.05,
            "volume": 900.0,
        }
    )


def _cfg(short_only: bool = True) -> dict:
    return {
        "data": {
            "include_futures_extras": True,
            "futures_extras": {
                "timeframe": "1h",
                "max_fill_gap_bars": 8,
                "funding": {"z_windows": [30, 90], "trend_window": 24, "abs_pctl_window": 180, "extreme_pctl": 0.95},
                "open_interest": {
                    "chg_windows": [1, 24],
                    "z_window": 30,
                    "oi_to_volume_window": 24,
                    "short_horizon_only": short_only,
                    "short_horizon_max": "30m",
                    "overlay": {"enabled": False, "recent_window_days": 30},
                },
            },
        }
    }


def test_stage38_oi_short_only_enforced_on_1h_runtime(tmp_path: Path) -> None:
    symbol = "BTC/USDT"
    bars_1h = _ohlcv("1h", 96)
    funding = pd.DataFrame({"timestamp": bars_1h["timestamp"], "funding_rate": [0.0001 for _ in range(len(bars_1h))]})
    oi = pd.DataFrame({"timestamp": bars_1h["timestamp"], "open_interest": [1_500_000.0 + i * 20.0 for i in range(len(bars_1h))]})
    save_derived_parquet(frame=funding, kind="funding", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    save_derived_parquet(frame=oi, kind="open_interest", symbol=symbol, timeframe="1h", data_dir=tmp_path)

    out = calculate_features(
        bars_1h,
        config=_cfg(short_only=True),
        symbol=symbol,
        timeframe="1h",
        derived_data_dir=tmp_path,
    )
    for col in OI_DEPENDENT_COLUMNS:
        assert col in out.columns
        assert out[col].isna().all()
    assert "oi_active" in out.columns
    assert out["oi_active"].astype(bool).sum() == 0
    usage = dict(out.attrs.get("oi_usage", {}))
    assert usage["short_only_enabled"] is True
    assert usage["timeframe_allowed"] is False
    assert usage["applied_mask"] is True
    assert usage["oi_active"] is False


def test_stage38_oi_short_only_allows_15m_runtime(tmp_path: Path) -> None:
    symbol = "ETH/USDT"
    bars_1h = _ohlcv("1h", 96)
    bars_15m = _ohlcv("15min", 96 * 4)
    funding = pd.DataFrame({"timestamp": bars_1h["timestamp"], "funding_rate": [0.0002 for _ in range(len(bars_1h))]})
    oi = pd.DataFrame({"timestamp": bars_1h["timestamp"], "open_interest": [1_700_000.0 + i * 25.0 for i in range(len(bars_1h))]})
    save_derived_parquet(frame=funding, kind="funding", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    save_derived_parquet(frame=oi, kind="open_interest", symbol=symbol, timeframe="1h", data_dir=tmp_path)

    out = calculate_features(
        bars_15m,
        config=_cfg(short_only=True),
        symbol=symbol,
        timeframe="15m",
        derived_data_dir=tmp_path,
    )
    assert "oi" in out.columns
    assert out["oi"].notna().sum() > 0
    usage = dict(out.attrs.get("oi_usage", {}))
    assert usage["short_only_enabled"] is True
    assert usage["timeframe_allowed"] is True
    assert usage["applied_mask"] is False
    assert usage["oi_active"] is True

