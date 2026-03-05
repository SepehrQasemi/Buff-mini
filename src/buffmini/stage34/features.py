"""OHLCV-only feature engineering for Stage-34 datasets."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


FeatureFn = Callable[[pd.DataFrame], pd.Series]


def compute_ohlcv_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy().sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    close = work["close"]
    high = work["high"]
    low = work["low"]
    open_ = work["open"]
    volume = work["volume"]
    prev_close = close.shift(1)

    ret_1 = close.pct_change()
    log_ret_1 = np.log(close / prev_close.replace(0.0, np.nan))
    ret_3 = close.pct_change(3)
    ret_6 = close.pct_change(6)
    ret_12 = close.pct_change(12)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = tr.rolling(14, min_periods=2).mean()
    atr_pct = atr_14 / close.replace(0.0, np.nan)

    hl_range = (high - low).replace(0.0, np.nan)
    body = (close - open_).abs()
    upper_wick = (high - pd.concat([open_, close], axis=1).max(axis=1)).clip(lower=0.0)
    lower_wick = (pd.concat([open_, close], axis=1).min(axis=1) - low).clip(lower=0.0)
    body_ratio = body / hl_range
    upper_wick_ratio = upper_wick / hl_range
    lower_wick_ratio = lower_wick / hl_range
    hl_range_pct = hl_range / close.replace(0.0, np.nan)

    vol_12 = ret_1.rolling(12, min_periods=4).std(ddof=0)
    vol_24 = ret_1.rolling(24, min_periods=8).std(ddof=0)
    vol_48 = ret_1.rolling(48, min_periods=12).std(ddof=0)

    volume_med_24 = volume.rolling(24, min_periods=8).median()
    volume_std_24 = volume.rolling(24, min_periods=8).std(ddof=0)
    volume_std_72 = volume.rolling(72, min_periods=12).std(ddof=0)
    volume_z_24 = (volume - volume_med_24) / volume_std_24.replace(0.0, np.nan)
    volume_z_72 = (volume - volume.rolling(72, min_periods=12).median()) / volume_std_72.replace(0.0, np.nan)
    volume_shock = volume / volume_med_24.replace(0.0, np.nan)

    ma_10 = close.rolling(10, min_periods=3).mean()
    ma_20 = close.rolling(20, min_periods=5).mean()
    ma_50 = close.rolling(50, min_periods=10).mean()
    ma_dist_20 = (close - ma_20) / close.replace(0.0, np.nan)
    ma_dist_50 = (close - ma_50) / close.replace(0.0, np.nan)
    ma_slope_20 = ma_20.pct_change(5)
    ma_slope_50 = ma_50.pct_change(10)

    rv_24 = (ret_1**2).rolling(24, min_periods=8).sum()
    rv_72 = (ret_1**2).rolling(72, min_periods=12).sum()
    compression_24 = vol_24 / vol_48.replace(0.0, np.nan)
    expansion_24 = vol_12 / vol_24.replace(0.0, np.nan)

    rolling_high_20 = high.rolling(20, min_periods=5).max().shift(1)
    rolling_low_20 = low.rolling(20, min_periods=5).min().shift(1)
    breakout_20 = close / rolling_high_20.replace(0.0, np.nan) - 1.0
    breakdown_20 = close / rolling_low_20.replace(0.0, np.nan) - 1.0
    std_20 = close.rolling(20, min_periods=5).std(ddof=0)
    meanrev_20 = (close - ma_20) / std_20.replace(0.0, np.nan)

    ts = pd.to_datetime(work.get("timestamp"), utc=True, errors="coerce")
    hour = ts.dt.hour.fillna(0).astype(float)
    dow = ts.dt.dayofweek.fillna(0).astype(float)
    tod_sin = np.sin(2.0 * np.pi * hour / 24.0)
    tod_cos = np.cos(2.0 * np.pi * hour / 24.0)
    dow_sin = np.sin(2.0 * np.pi * dow / 7.0)
    dow_cos = np.cos(2.0 * np.pi * dow / 7.0)

    feats = pd.DataFrame(
        {
            "timestamp": ts,
            "ret_1": ret_1,
            "log_ret_1": log_ret_1,
            "ret_3": ret_3,
            "ret_6": ret_6,
            "ret_12": ret_12,
            "atr_14": atr_14,
            "atr_pct": atr_pct,
            "hl_range_pct": hl_range_pct,
            "body_ratio": body_ratio,
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "vol_12": vol_12,
            "vol_24": vol_24,
            "vol_48": vol_48,
            "volume_z_24": volume_z_24,
            "volume_z_72": volume_z_72,
            "volume_shock": volume_shock,
            "ma_10": ma_10,
            "ma_20": ma_20,
            "ma_50": ma_50,
            "ma_dist_20": ma_dist_20,
            "ma_dist_50": ma_dist_50,
            "ma_slope_20": ma_slope_20,
            "ma_slope_50": ma_slope_50,
            "rv_24": rv_24,
            "rv_72": rv_72,
            "compression_24": compression_24,
            "expansion_24": expansion_24,
            "breakout_20": breakout_20,
            "breakdown_20": breakdown_20,
            "meanrev_20": meanrev_20,
            "tod_sin": tod_sin,
            "tod_cos": tod_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
        }
    )
    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats


def feature_columns(max_features: int = 120) -> list[str]:
    cols = [
        "ret_1",
        "log_ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "atr_14",
        "atr_pct",
        "hl_range_pct",
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "vol_12",
        "vol_24",
        "vol_48",
        "volume_z_24",
        "volume_z_72",
        "volume_shock",
        "ma_10",
        "ma_20",
        "ma_50",
        "ma_dist_20",
        "ma_dist_50",
        "ma_slope_20",
        "ma_slope_50",
        "rv_24",
        "rv_72",
        "compression_24",
        "expansion_24",
        "breakout_20",
        "breakdown_20",
        "meanrev_20",
        "tod_sin",
        "tod_cos",
        "dow_sin",
        "dow_cos",
    ]
    return cols[: int(max(1, max_features))]


def feature_function_registry(max_features: int = 120) -> dict[str, FeatureFn]:
    cols = feature_columns(max_features=max_features)
    registry: dict[str, FeatureFn] = {}
    for col in cols:
        registry[col] = _feature_accessor(col)
    return registry


def _feature_accessor(col: str) -> FeatureFn:
    def _fn(frame: pd.DataFrame) -> pd.Series:
        feats = compute_ohlcv_features(frame)
        return pd.to_numeric(feats.get(col, np.nan), errors="coerce")

    return _fn

