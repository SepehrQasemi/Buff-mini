"""Stage-9.3 futures feature leakage tests."""

from __future__ import annotations

import pandas as pd

from buffmini.data.features import calculate_features, registered_feature_columns
from buffmini.data.features_futures import build_all_futures_features, registered_futures_feature_columns, synthetic_futures_extras
from buffmini.validation.leakage_harness import run_registered_features_harness, synthetic_ohlcv


def _cfg_include_extras() -> dict:
    return {
        "data": {
            "include_futures_extras": True,
            "futures_extras": {
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "timeframe": "1h",
                "max_fill_gap_bars": 8,
                "funding": {
                    "z_windows": [30, 90],
                    "trend_window": 24,
                    "abs_pctl_window": 180,
                    "extreme_pctl": 0.95,
                },
                "open_interest": {
                    "chg_windows": [1, 24],
                    "z_window": 30,
                    "oi_to_volume_window": 24,
                },
            },
        }
    }


def test_calculate_features_with_futures_extras_columns_present() -> None:
    frame = synthetic_ohlcv(rows=420, seed=42)
    features = calculate_features(
        frame,
        config=_cfg_include_extras(),
        symbol="BTC/USDT",
        timeframe="1h",
        _synthetic_extras_for_tests=True,
    )

    for col in registered_futures_feature_columns():
        assert col in features.columns


def test_futures_features_future_shock_does_not_change_past() -> None:
    ohlcv = synthetic_ohlcv(rows=420, seed=7)
    funding, oi = synthetic_futures_extras(ohlcv)
    base = build_all_futures_features(ohlcv, funding, oi, config=_cfg_include_extras()["data"]["futures_extras"])

    cutoff = 300
    shocked_funding = funding.copy()
    shocked_oi = oi.copy()
    shocked_funding.loc[cutoff + 1 :, "funding_rate"] = shocked_funding.loc[cutoff + 1 :, "funding_rate"] * 50.0
    shocked_oi.loc[cutoff + 1 :, "open_interest"] = shocked_oi.loc[cutoff + 1 :, "open_interest"] * 5.0

    recomputed = build_all_futures_features(
        ohlcv,
        shocked_funding,
        shocked_oi,
        config=_cfg_include_extras()["data"]["futures_extras"],
    )

    cols = [
        "funding_rate",
        "funding_z_30",
        "funding_z_90",
        "funding_trend_24",
        "funding_abs_pctl_180d",
        "oi",
        "oi_chg_1",
        "oi_chg_24",
        "oi_z_30",
        "oi_to_volume",
        "oi_accel",
        "crowd_long_risk",
        "crowd_short_risk",
        "leverage_building",
    ]
    for col in cols:
        pd.testing.assert_series_equal(
            base.loc[:cutoff, col].reset_index(drop=True),
            recomputed.loc[:cutoff, col].reset_index(drop=True),
            check_names=False,
        )


def test_leakage_harness_checks_futures_features_when_enabled() -> None:
    base_result = run_registered_features_harness(rows=320, seed=42, shock_index=260, warmup_max=220)
    futures_result = run_registered_features_harness(
        rows=420,
        seed=42,
        shock_index=360,
        warmup_max=220,
        include_futures_extras=True,
    )

    assert futures_result["features_checked"] > base_result["features_checked"]
    assert futures_result["features_checked"] == len(registered_feature_columns(include_futures_extras=True))
    assert futures_result["leaks_found"] == 0
