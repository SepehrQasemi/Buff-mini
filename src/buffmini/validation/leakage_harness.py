"""Stage-8.3 automated no-future-leakage harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from buffmini.data.features import calculate_features, registered_feature_columns


@dataclass(frozen=True)
class LeakageCheck:
    """Per-feature leakage check result."""

    feature: str
    leaked: bool
    max_abs_diff: float
    checked_points: int


FeatureFunction = Callable[[pd.DataFrame], pd.Series]


def synthetic_ohlcv(rows: int = 360, seed: int = 42) -> pd.DataFrame:
    """Build deterministic synthetic OHLCV for leakage checks."""

    rng = np.random.default_rng(int(seed))
    timestamps = pd.date_range("2025-01-01", periods=int(rows), freq="h", tz="UTC")
    base = 100.0 + np.cumsum(rng.normal(0.0, 0.7, size=int(rows)))
    high = base + rng.uniform(0.05, 0.9, size=int(rows))
    low = base - rng.uniform(0.05, 0.9, size=int(rows))
    volume = rng.uniform(1000.0, 1500.0, size=int(rows))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": base,
            "high": high,
            "low": low,
            "close": base,
            "volume": volume,
        }
    )


def apply_future_shock(frame: pd.DataFrame, shock_index: int) -> pd.DataFrame:
    """Apply deterministic shock only on future rows (> shock_index)."""

    shocked = frame.copy()
    idx = int(shock_index)
    mask = shocked.index > idx
    shocked.loc[mask, "close"] = shocked.loc[mask, "close"] * 8.0
    shocked.loc[mask, "high"] = shocked.loc[mask, "high"] * 8.5
    shocked.loc[mask, "low"] = shocked.loc[mask, "low"] * 0.2
    shocked.loc[mask, "open"] = shocked.loc[mask, "open"] * 7.5
    return shocked


def run_feature_harness(
    frame: pd.DataFrame,
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    feature_columns: list[str],
    shock_index: int,
    warmup_max: int,
    tol: float = 1e-12,
) -> dict[str, object]:
    """Run future-shock leakage check for a dataframe feature function."""

    if int(shock_index) <= 1:
        raise ValueError("shock_index must be > 1")
    if int(warmup_max) < 0:
        raise ValueError("warmup_max must be >= 0")

    base = feature_fn(frame.copy())
    shocked = feature_fn(apply_future_shock(frame, int(shock_index)))

    safe_end = int(shock_index) - int(warmup_max) - 1
    if safe_end < 0:
        safe_end = 0

    checks: list[LeakageCheck] = []
    for col in feature_columns:
        if col not in base.columns or col not in shocked.columns:
            raise ValueError(f"feature column missing in harness output: {col}")
        leaked, max_abs_diff, checked = compare_series_no_future_leakage(
            baseline=base[col],
            shocked=shocked[col],
            safe_end=safe_end,
            tol=float(tol),
        )
        checks.append(
            LeakageCheck(
                feature=str(col),
                leaked=bool(leaked),
                max_abs_diff=float(max_abs_diff),
                checked_points=int(checked),
            )
        )

    leaked_features = sorted([check.feature for check in checks if check.leaked])
    return {
        "features_checked": int(len(checks)),
        "leaks_found": int(len(leaked_features)),
        "leaked_features": leaked_features,
        "safe_region_end": int(safe_end),
        "shock_index": int(shock_index),
        "checks": [
            {
                "feature": check.feature,
                "leaked": bool(check.leaked),
                "max_abs_diff": float(check.max_abs_diff),
                "checked_points": int(check.checked_points),
            }
            for check in checks
        ],
    }


def run_feature_functions_harness(
    frame: pd.DataFrame,
    feature_funcs: dict[str, FeatureFunction],
    shock_index: int,
    warmup_max: int,
    tol: float = 1e-12,
) -> dict[str, object]:
    """Run leakage checks for a feature-function registry."""

    if not feature_funcs:
        raise ValueError("feature_funcs must be non-empty")

    baseline = frame.copy()
    shocked_frame = apply_future_shock(frame, int(shock_index))
    safe_end = int(shock_index) - int(warmup_max) - 1
    if safe_end < 0:
        safe_end = 0

    checks: list[LeakageCheck] = []
    for name in sorted(feature_funcs):
        func = feature_funcs[name]
        base_series = pd.Series(func(baseline.copy()))
        shocked_series = pd.Series(func(shocked_frame.copy()))
        leaked, max_abs_diff, checked = compare_series_no_future_leakage(
            baseline=base_series,
            shocked=shocked_series,
            safe_end=safe_end,
            tol=float(tol),
        )
        checks.append(
            LeakageCheck(
                feature=str(name),
                leaked=bool(leaked),
                max_abs_diff=float(max_abs_diff),
                checked_points=int(checked),
            )
        )

    leaked_features = sorted([check.feature for check in checks if check.leaked])
    return {
        "features_checked": int(len(checks)),
        "leaks_found": int(len(leaked_features)),
        "leaked_features": leaked_features,
        "safe_region_end": int(safe_end),
        "shock_index": int(shock_index),
        "checks": [
            {
                "feature": check.feature,
                "leaked": bool(check.leaked),
                "max_abs_diff": float(check.max_abs_diff),
                "checked_points": int(check.checked_points),
            }
            for check in checks
        ],
    }


def run_registered_features_harness(
    rows: int = 360,
    seed: int = 42,
    shock_index: int = 280,
    warmup_max: int = 220,
    tol: float = 1e-12,
) -> dict[str, object]:
    """Run leakage harness against all registered project features."""

    frame = synthetic_ohlcv(rows=int(rows), seed=int(seed))
    return run_feature_harness(
        frame=frame,
        feature_fn=calculate_features,
        feature_columns=registered_feature_columns(),
        shock_index=int(shock_index),
        warmup_max=int(warmup_max),
        tol=float(tol),
    )


def compare_series_no_future_leakage(
    baseline: pd.Series,
    shocked: pd.Series,
    safe_end: int,
    tol: float = 1e-12,
) -> tuple[bool, float, int]:
    """Compare two series in pre-shock safe region and detect leakage."""

    end = int(min(len(baseline), len(shocked), int(safe_end) + 1))
    if end <= 0:
        return False, 0.0, 0

    left = pd.Series(baseline.iloc[:end])
    right = pd.Series(shocked.iloc[:end])

    # Numeric series: tolerance-based comparison with NaN treated as equal.
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        left_vals = pd.to_numeric(left, errors="coerce").astype(float).to_numpy()
        right_vals = pd.to_numeric(right, errors="coerce").astype(float).to_numpy()
        nan_mask = np.isnan(left_vals) & np.isnan(right_vals)
        equal_mask = np.isclose(left_vals, right_vals, atol=float(tol), rtol=0.0, equal_nan=True)
        combined_equal = equal_mask | nan_mask
        leaked = bool(not np.all(combined_equal))
        diff = np.abs(left_vals - right_vals)
        finite = diff[np.isfinite(diff)]
        max_abs_diff = float(finite.max()) if finite.size else 0.0
        return leaked, max_abs_diff, int(end)

    left_obj = left.astype("object")
    right_obj = right.astype("object")
    equal = (left_obj == right_obj) | (left_obj.isna() & right_obj.isna())
    leaked = bool(not bool(equal.all()))
    return leaked, 1.0 if leaked else 0.0, int(end)
