"""Stage-8.3 leakage harness tests."""

from __future__ import annotations

import pandas as pd

from buffmini.data.features import registered_feature_columns
from buffmini.validation.leakage_harness import (
    run_feature_functions_harness,
    run_registered_features_harness,
    synthetic_ohlcv,
)


def test_harness_detects_deliberate_leakage() -> None:
    frame = synthetic_ohlcv(rows=140, seed=7)

    funcs = {
        "safe_lag": lambda df: pd.Series(df["close"], dtype=float).shift(1),
        "leaky_future": lambda df: pd.Series(df["close"], dtype=float).rolling(30, center=True, min_periods=1).mean(),
    }
    result = run_feature_functions_harness(
        frame=frame,
        feature_funcs=funcs,
        shock_index=100,
        warmup_max=10,
        tol=1e-12,
    )

    assert result["features_checked"] == 2
    assert result["leaks_found"] == 1
    assert result["leaked_features"] == ["leaky_future"]


def test_registered_features_pass_harness() -> None:
    result = run_registered_features_harness(rows=420, seed=42, shock_index=360, warmup_max=252)
    assert result["features_checked"] == len(registered_feature_columns())
    assert result["leaks_found"] == 0
    assert result["leaked_features"] == []


def test_harness_runtime_small_input() -> None:
    result = run_registered_features_harness(rows=280, seed=5, shock_index=230, warmup_max=200)
    assert result["features_checked"] > 0
    assert isinstance(result["checks"], list)
