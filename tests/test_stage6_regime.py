"""Stage-6 regime classifier tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.data.features import calculate_features


def _frame(rows: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ts = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=rows))
    high = close + rng.uniform(0.05, 1.25, size=rows)
    low = close - rng.uniform(0.05, 1.25, size=rows)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1000.0, 2000.0, size=rows),
        }
    )


def test_stage6_regime_columns_and_non_null_after_warmup() -> None:
    features = calculate_features(_frame())
    for column in ["trend_strength", "atr_percentile_252", "regime"]:
        assert column in features.columns
    warm = features.iloc[300:].copy()
    assert warm["trend_strength"].notna().all()
    assert warm["atr_percentile_252"].notna().all()
    assert warm["regime"].notna().all()


def test_stage6_regime_no_future_leakage_future_shock() -> None:
    base = _frame()
    baseline = calculate_features(base)

    shocked = base.copy()
    cutoff = 320
    shocked.loc[cutoff + 1 :, "close"] = shocked.loc[cutoff + 1 :, "close"] * 1.35
    shocked.loc[cutoff + 1 :, "high"] = shocked.loc[cutoff + 1 :, "high"] * 1.50
    shocked.loc[cutoff + 1 :, "low"] = shocked.loc[cutoff + 1 :, "low"] * 0.70
    recomputed = calculate_features(shocked)

    for column in ["trend_strength", "atr_percentile_252", "regime"]:
        pd.testing.assert_series_equal(
            baseline.loc[:cutoff, column].reset_index(drop=True),
            recomputed.loc[:cutoff, column].reset_index(drop=True),
            check_names=False,
        )


def test_stage6_regime_deterministic_for_fixed_input() -> None:
    frame = _frame()
    first = calculate_features(frame)
    second = calculate_features(frame)
    pd.testing.assert_series_equal(first["regime"], second["regime"], check_names=False)

