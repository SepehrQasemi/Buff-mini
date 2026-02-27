"""Deterministic Stage-6 regime classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REGIME_TREND = "TREND"
REGIME_RANGE = "RANGE"
REGIME_VOL_EXPANSION = "VOL_EXPANSION"


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Return rolling percentile-rank of the latest value in each window.

    The result at index ``t`` only uses values up to ``t`` (no future leakage).
    """

    if int(window) < 1:
        raise ValueError("window must be >= 1")

    values = pd.Series(series, dtype=float)

    def _rank_last(frame_values: np.ndarray) -> float:
        arr = np.asarray(frame_values, dtype=float)
        if arr.size == 0 or np.isnan(arr).all():
            return float("nan")
        last = arr[-1]
        # Fraction of observations <= current value in the trailing window.
        return float(np.mean(arr <= last))

    return values.rolling(window=int(window), min_periods=int(window)).apply(_rank_last, raw=True)


def classify_regime_from_features(
    atr_percentile: pd.Series,
    trend_strength: pd.Series,
    vol_expansion_threshold: float = 0.80,
    trend_threshold: float = 0.010,
    range_atr_threshold: float = 0.40,
) -> pd.Series:
    """Classify each bar as TREND/RANGE/VOL_EXPANSION using deterministic rules."""

    atr_pct = pd.Series(atr_percentile, dtype=float)
    trend = pd.Series(trend_strength, dtype=float)

    vol_mask = atr_pct >= float(vol_expansion_threshold)
    trend_mask = (atr_pct < float(vol_expansion_threshold)) & (trend >= float(trend_threshold))
    range_mask = (atr_pct <= float(range_atr_threshold)) | (trend < float(trend_threshold))

    regime = np.where(vol_mask, REGIME_VOL_EXPANSION, np.where(trend_mask, REGIME_TREND, REGIME_RANGE))
    # Keep explicit RANGE fallback for all non-vol/non-trend rows.
    regime = np.where(~vol_mask & ~trend_mask & range_mask, REGIME_RANGE, regime)
    return pd.Series(regime, index=atr_pct.index, dtype="object")


def attach_regime_columns(
    frame: pd.DataFrame,
    atr_percentile_window: int = 252,
    vol_expansion_threshold: float = 0.80,
    trend_threshold: float = 0.010,
    range_atr_threshold: float = 0.40,
) -> pd.DataFrame:
    """Attach Stage-6 regime helper columns to a feature frame.

    Required base columns: ``close``, ``atr_14``, ``ema_50``, ``ema_200``.
    """

    required = {"close", "atr_14", "ema_50", "ema_200"}
    missing = required.difference(frame.columns)
    if missing:
        msg = f"Missing regime input columns: {sorted(missing)}"
        raise ValueError(msg)

    data = frame.copy()
    close = pd.Series(data["close"], dtype=float).replace(0.0, np.nan)

    if "trend_strength" not in data.columns:
        data["trend_strength"] = (pd.Series(data["ema_50"], dtype=float) - pd.Series(data["ema_200"], dtype=float)).abs() / close

    if "atr_percentile_252" not in data.columns:
        data["atr_percentile_252"] = rolling_percentile_rank(
            series=pd.Series(data["atr_14"], dtype=float),
            window=int(atr_percentile_window),
        )

    data["regime"] = classify_regime_from_features(
        atr_percentile=pd.Series(data["atr_percentile_252"], dtype=float),
        trend_strength=pd.Series(data["trend_strength"], dtype=float),
        vol_expansion_threshold=float(vol_expansion_threshold),
        trend_threshold=float(trend_threshold),
        range_atr_threshold=float(range_atr_threshold),
    )
    return data


def regime_distribution_percent(frame: pd.DataFrame, column: str = "regime") -> dict[str, float]:
    """Return regime distribution percentages for reporting."""

    if column not in frame.columns or frame.empty:
        return {
            REGIME_TREND: 0.0,
            REGIME_RANGE: 0.0,
            REGIME_VOL_EXPANSION: 0.0,
        }
    counts = frame[column].astype(str).value_counts(normalize=True)
    return {
        REGIME_TREND: float(counts.get(REGIME_TREND, 0.0) * 100.0),
        REGIME_RANGE: float(counts.get(REGIME_RANGE, 0.0) * 100.0),
        REGIME_VOL_EXPANSION: float(counts.get(REGIME_VOL_EXPANSION, 0.0) * 100.0),
    }


def stage6_regime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fetch normalized Stage-6 regime config payload."""

    return (
        config.get("evaluation", {})
        .get("stage6", {})
        .get("regime", {})
    )

