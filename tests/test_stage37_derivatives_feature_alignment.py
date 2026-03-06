from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.data.derived_store import save_derived_parquet
from buffmini.data.features import calculate_features
from buffmini.validation.leakage_harness import compare_series_no_future_leakage


def _ohlcv(rows: int = 96) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01T00:00:00Z", periods=rows, freq="1h", tz="UTC")
    base = 100.0 + (pd.Series(range(rows), dtype=float) * 0.25)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.1,
            "volume": 1000.0 + (pd.Series(range(rows), dtype=float) * 3.0),
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
                    "short_horizon_only": False,
                    "short_horizon_max": "30m",
                },
                "taker_buy_sell": {"enabled": True, "z_window": 48, "burst_z": 1.5},
                "long_short_ratio": {"enabled": True, "z_window": 48, "extreme_z": 1.5},
            },
        }
    }


def test_stage37_derivatives_features_align_and_exist(tmp_path: Path) -> None:
    bars = _ohlcv(80)
    symbol = "BTC/USDT"
    funding = pd.DataFrame({"timestamp": bars["timestamp"].iloc[::3], "funding_rate": [0.0001 + (i * 0.00001) for i in range(27)]})
    oi = pd.DataFrame({"timestamp": bars["timestamp"].iloc[::3], "open_interest": [1_000_000.0 + (i * 500.0) for i in range(27)]})
    taker = pd.DataFrame(
        {
            "timestamp": bars["timestamp"].iloc[::2],
            "taker_buy_volume": [2000.0 + i for i in range(40)],
            "taker_sell_volume": [1980.0 + i for i in range(40)],
            "taker_buy_sell_ratio": [1.0 + (i * 0.001) for i in range(40)],
        }
    )
    ls = pd.DataFrame(
        {
            "timestamp": bars["timestamp"].iloc[::2],
            "long_short_ratio": [1.0 + (i * 0.002) for i in range(40)],
            "long_account_ratio": [0.52 for _ in range(40)],
            "short_account_ratio": [0.48 for _ in range(40)],
        }
    )
    save_derived_parquet(frame=funding, kind="funding", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    save_derived_parquet(frame=oi, kind="open_interest", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    save_derived_parquet(frame=taker, kind="taker_buy_sell", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    save_derived_parquet(frame=ls, kind="long_short_ratio", symbol=symbol, timeframe="1h", data_dir=tmp_path)

    out = calculate_features(
        bars,
        config=_cfg(),
        symbol=symbol,
        timeframe="1h",
        derived_data_dir=tmp_path,
    )
    for col in (
        "funding_zscore",
        "funding_regime",
        "funding_shock",
        "taker_imbalance",
        "taker_burst_flag",
        "ls_ratio_level",
        "ls_ratio_zscore",
        "price_ratio_divergence",
    ):
        assert col in out.columns
    assert len(out) == len(bars)


def test_stage37_derivatives_no_future_leakage(tmp_path: Path) -> None:
    bars = _ohlcv(90)
    symbol = "ETH/USDT"
    base_taker = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "taker_buy_volume": [3000.0 + i for i in range(90)],
            "taker_sell_volume": [2900.0 + i for i in range(90)],
            "taker_buy_sell_ratio": [1.03 for _ in range(90)],
        }
    )
    base_ls = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "long_short_ratio": [1.0 + (0.01 * ((i % 8) - 4)) for i in range(90)],
            "long_account_ratio": [0.51 for _ in range(90)],
            "short_account_ratio": [0.49 for _ in range(90)],
        }
    )
    funding = pd.DataFrame({"timestamp": bars["timestamp"], "funding_rate": [0.0001 for _ in range(90)]})
    oi = pd.DataFrame({"timestamp": bars["timestamp"], "open_interest": [1_000_000.0 + (i * 10.0) for i in range(90)]})
    save_derived_parquet(frame=funding, kind="funding", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    save_derived_parquet(frame=oi, kind="open_interest", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    save_derived_parquet(frame=base_taker, kind="taker_buy_sell", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    save_derived_parquet(frame=base_ls, kind="long_short_ratio", symbol=symbol, timeframe="1h", data_dir=tmp_path)

    baseline = calculate_features(
        bars,
        config=_cfg(),
        symbol=symbol,
        timeframe="1h",
        derived_data_dir=tmp_path,
    )

    shocked_taker = base_taker.copy()
    shocked_ls = base_ls.copy()
    shocked_taker.loc[65:, "taker_buy_volume"] = shocked_taker.loc[65:, "taker_buy_volume"] * 5.0
    shocked_ls.loc[65:, "long_short_ratio"] = shocked_ls.loc[65:, "long_short_ratio"] * 2.0
    save_derived_parquet(frame=shocked_taker, kind="taker_buy_sell", symbol=symbol, timeframe="1h", data_dir=tmp_path)
    save_derived_parquet(frame=shocked_ls, kind="long_short_ratio", symbol=symbol, timeframe="1h", data_dir=tmp_path)

    after = calculate_features(
        bars,
        config=_cfg(),
        symbol=symbol,
        timeframe="1h",
        derived_data_dir=tmp_path,
    )

    leaked_imbalance, _, _ = compare_series_no_future_leakage(
        baseline=baseline["taker_imbalance"],
        shocked=after["taker_imbalance"],
        safe_end=50,
        tol=1e-12,
    )
    leaked_ls, _, _ = compare_series_no_future_leakage(
        baseline=baseline["ls_ratio_level"],
        shocked=after["ls_ratio_level"],
        safe_end=50,
        tol=1e-12,
    )
    assert leaked_imbalance is False
    assert leaked_ls is False
