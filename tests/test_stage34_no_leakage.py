from __future__ import annotations

import pandas as pd

from buffmini.stage34.features import feature_function_registry
from buffmini.validation.leakage_harness import run_feature_functions_harness, synthetic_ohlcv


def test_stage34_feature_registry_no_future_leakage() -> None:
    frame = synthetic_ohlcv(rows=420, seed=42)
    funcs = feature_function_registry(max_features=50)
    result = run_feature_functions_harness(
        frame=frame,
        feature_funcs=funcs,
        shock_index=320,
        warmup_max=150,
    )
    assert result["features_checked"] >= 10
    assert result["leaks_found"] == 0


def test_stage34_leaky_feature_probe_is_detected() -> None:
    frame = synthetic_ohlcv(rows=420, seed=7)
    funcs = {
        "safe": lambda df: pd.to_numeric(df["close"], errors="coerce").rolling(10, min_periods=2).mean(),
        "leaky": lambda df: pd.to_numeric(df["close"], errors="coerce").shift(-220),
    }
    result = run_feature_functions_harness(
        frame=frame,
        feature_funcs=funcs,
        shock_index=320,
        warmup_max=150,
    )
    assert result["features_checked"] == 2
    assert result["leaks_found"] >= 1
    assert "leaky" in result["leaked_features"]
