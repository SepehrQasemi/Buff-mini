"""Reusable MTF feature pack computation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FEATURE_PACK_VERSION = "stage11_mtf_pack_v1"


def compute_feature_pack(
    frame: pd.DataFrame,
    layer_name: str,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute leakage-safe feature pack for one timeframe layer."""

    cfg = dict(params or {})
    ema_fast = int(cfg.get("ema_fast", 50))
    ema_slow = int(cfg.get("ema_slow", 200))
    ema_slope_lookback = int(cfg.get("ema_slope_lookback", 24))
    atr_window = int(cfg.get("atr_window", 14))
    atr_rank_window = int(cfg.get("atr_rank_window", 252))
    bb_window = int(cfg.get("bb_window", 20))
    bb_std_k = float(cfg.get("bb_std_k", 2.0))
    volume_z_window = int(cfg.get("volume_z_window", 120))

    required = {"open", "high", "low", "close", "volume", "ts_open", "ts_close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for MTF feature pack: {sorted(missing)}")

    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    high = pd.to_numeric(out["high"], errors="coerce").astype(float)
    low = pd.to_numeric(out["low"], errors="coerce").astype(float)
    volume = pd.to_numeric(out["volume"], errors="coerce").astype(float)

    ema_fast_series = close.ewm(span=ema_fast, adjust=False, min_periods=ema_fast).mean()
    ema_slow_series = close.ewm(span=ema_slow, adjust=False, min_periods=ema_slow).mean()
    ema_slope = (ema_fast_series / ema_fast_series.shift(ema_slope_lookback)) - 1.0

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_window, min_periods=atr_window).mean()
    atr_pct = atr / close.replace(0.0, np.nan)
    atr_pct_rank = _rolling_percentile(atr_pct, window=atr_rank_window)

    bb_mid = close.rolling(window=bb_window, min_periods=bb_window).mean()
    bb_std = close.rolling(window=bb_window, min_periods=bb_window).std(ddof=0)
    bb_upper = bb_mid + bb_std_k * bb_std
    bb_lower = bb_mid - bb_std_k * bb_std
    bb_bandwidth = (bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)

    volume_z = _rolling_zscore(volume, window=volume_z_window)

    prefix = f"{layer_name}__"
    features = pd.DataFrame(
        {
            "ts_open": pd.to_datetime(out["ts_open"], utc=True, errors="coerce"),
            "ts_close": pd.to_datetime(out["ts_close"], utc=True, errors="coerce"),
            f"{prefix}ema_50": ema_fast_series,
            f"{prefix}ema_200": ema_slow_series,
            f"{prefix}ema_slope_50": ema_slope,
            f"{prefix}atr_14": atr,
            f"{prefix}atr_pct": atr_pct,
            f"{prefix}atr_pct_rank_252": atr_pct_rank,
            f"{prefix}bb_mid_20": bb_mid,
            f"{prefix}bb_upper_20_2": bb_upper,
            f"{prefix}bb_lower_20_2": bb_lower,
            f"{prefix}bb_bandwidth_20": bb_bandwidth,
            f"{prefix}volume_z_120": volume_z,
        }
    )
    return features.reset_index(drop=True)


def feature_pack_columns(layer_name: str) -> list[str]:
    prefix = f"{layer_name}__"
    return [
        f"{prefix}ema_50",
        f"{prefix}ema_200",
        f"{prefix}ema_slope_50",
        f"{prefix}atr_14",
        f"{prefix}atr_pct",
        f"{prefix}atr_pct_rank_252",
        f"{prefix}bb_mid_20",
        f"{prefix}bb_upper_20_2",
        f"{prefix}bb_lower_20_2",
        f"{prefix}bb_bandwidth_20",
        f"{prefix}volume_z_120",
    ]


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)

    def _rank(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or np.isnan(arr).all():
            return np.nan
        last = arr[-1]
        return float(np.mean(arr <= last))

    return s.rolling(window=window, min_periods=window).apply(_rank, raw=True)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mu = s.rolling(window=window, min_periods=window).mean()
    sigma = s.rolling(window=window, min_periods=window).std(ddof=0).replace(0.0, np.nan)
    return (s - mu) / sigma

