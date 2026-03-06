from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.data.derived_store import save_derived_parquet
from buffmini.data.features import calculate_features
from buffmini.stage38.audit import oi_runtime_usage
from buffmini.stage9.oi_overlay import OI_DEPENDENT_COLUMNS


def _bars(rows: int = 64) -> pd.DataFrame:
    ts = pd.date_range("2025-08-01T00:00:00Z", periods=rows, freq="1h", tz="UTC")
    base = 200.0 + (pd.Series(range(rows), dtype=float) * 0.05)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": base,
            "high": base + 0.15,
            "low": base - 0.15,
            "close": base + 0.03,
            "volume": 850.0,
        }
    )


def _cfg() -> dict:
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
                    "short_horizon_only": True,
                    "short_horizon_max": "30m",
                    "overlay": {"enabled": False, "recent_window_days": 30},
                },
            },
        }
    }


def test_stage38_oi_usage_report_matches_feature_runtime(tmp_path: Path) -> None:
    symbol = "BTC/USDT"
    bars = _bars(80)
    funding = pd.DataFrame({"timestamp": bars["timestamp"], "funding_rate": [0.0001 for _ in range(len(bars))]})
    oi = pd.DataFrame({"timestamp": bars["timestamp"], "open_interest": [1_300_000.0 + i * 10.0 for i in range(len(bars))]})
    save_derived_parquet(frame=funding, kind="funding", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    save_derived_parquet(frame=oi, kind="open_interest", symbol=symbol, timeframe="1h", data_dir=tmp_path)

    out = calculate_features(
        bars,
        config=_cfg(),
        symbol=symbol,
        timeframe="1h",
        derived_data_dir=tmp_path,
    )
    runtime = oi_runtime_usage(
        frame=out,
        oi_columns=list(OI_DEPENDENT_COLUMNS),
        timeframe="1h",
        short_horizon_max="30m",
        short_only_enabled=True,
    )
    attrs = dict(out.attrs.get("oi_usage", {}))
    assert bool(runtime["oi_active_runtime"]) == bool(attrs.get("oi_active", False))
    assert bool(runtime["timeframe_allowed"]) == bool(attrs.get("timeframe_allowed", False))

