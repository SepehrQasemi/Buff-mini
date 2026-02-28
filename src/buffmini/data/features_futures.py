"""Leakage-safe funding/open-interest feature builders for Stage-9."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.data.futures_extras import align_funding_to_ohlcv, align_open_interest_to_ohlcv

FUTURES_FEATURE_COLUMNS: tuple[str, ...] = (
    "funding_rate",
    "funding_z_30",
    "funding_z_90",
    "funding_trend_24",
    "funding_abs_pctl_180d",
    "funding_extreme_pos",
    "funding_extreme_neg",
    "oi",
    "oi_chg_1",
    "oi_chg_24",
    "oi_z_30",
    "oi_to_volume",
    "oi_accel",
    "crowd_long_risk",
    "crowd_short_risk",
    "leverage_building",
)


def registered_futures_feature_columns() -> list[str]:
    """Return canonical Stage-9 futures feature columns."""

    return list(FUTURES_FEATURE_COLUMNS)


def build_funding_features(
    df_ohlcv: pd.DataFrame,
    df_funding: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    timeframe: str = "1h",
    max_fill_gap_bars: int = 8,
) -> pd.DataFrame:
    """Build leakage-safe funding features aligned to OHLCV bar timestamps."""

    cfg = settings or {}
    z_windows = [int(value) for value in cfg.get("z_windows", [30, 90])]
    trend_window = int(cfg.get("trend_window", 24))
    abs_pctl_window = int(cfg.get("abs_pctl_window", 4320))
    extreme_pctl = float(cfg.get("extreme_pctl", 0.95))

    aligned = _normalize_funding_input(df_ohlcv=df_ohlcv, df_funding=df_funding, timeframe=timeframe)
    rate = pd.to_numeric(aligned["funding_rate"], errors="coerce").astype(float)
    if int(max_fill_gap_bars) >= 0:
        rate = rate.ffill(limit=int(max_fill_gap_bars))

    out = pd.DataFrame({"timestamp": pd.to_datetime(df_ohlcv["timestamp"], utc=True), "funding_rate": rate})

    for window in z_windows:
        mu = rate.rolling(window=window, min_periods=window).mean()
        sigma = rate.rolling(window=window, min_periods=window).std(ddof=0)
        out[f"funding_z_{window}"] = (rate - mu) / sigma.replace(0.0, np.nan)

    out[f"funding_trend_{trend_window}"] = rate.diff(int(trend_window))

    abs_rate = rate.abs()
    out["funding_abs_pctl_180d"] = _rolling_percentile(abs_rate, window=abs_pctl_window)
    out["funding_extreme_pos"] = (
        (out["funding_abs_pctl_180d"] >= float(extreme_pctl)) & (rate > 0.0)
    ).astype(int)
    out["funding_extreme_neg"] = (
        (out["funding_abs_pctl_180d"] >= float(extreme_pctl)) & (rate < 0.0)
    ).astype(int)
    return out


def build_oi_features(
    df_ohlcv: pd.DataFrame,
    df_oi: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    timeframe: str = "1h",
    max_fill_gap_bars: int = 8,
) -> pd.DataFrame:
    """Build leakage-safe open-interest features aligned to OHLCV bar timestamps."""

    cfg = settings or {}
    chg_windows = [int(value) for value in cfg.get("chg_windows", [1, 24])]
    z_window = int(cfg.get("z_window", 30))
    oi_to_volume_window = int(cfg.get("oi_to_volume_window", 24))

    aligned = _normalize_oi_input(df_ohlcv=df_ohlcv, df_oi=df_oi, timeframe=timeframe)
    oi = pd.to_numeric(aligned["open_interest"], errors="coerce").astype(float)
    if int(max_fill_gap_bars) >= 0:
        oi = oi.ffill(limit=int(max_fill_gap_bars))

    out = pd.DataFrame({"timestamp": pd.to_datetime(df_ohlcv["timestamp"], utc=True), "oi": oi})

    if 1 in chg_windows:
        out["oi_chg_1"] = oi.diff(1)
    else:
        out["oi_chg_1"] = oi.diff(1)

    if 24 in chg_windows:
        out["oi_chg_24"] = oi.diff(24)
    else:
        out["oi_chg_24"] = oi.diff(int(max(chg_windows)))

    mu = oi.rolling(window=z_window, min_periods=z_window).mean()
    sigma = oi.rolling(window=z_window, min_periods=z_window).std(ddof=0)
    out["oi_z_30"] = (oi - mu) / sigma.replace(0.0, np.nan)

    volume_roll = pd.to_numeric(df_ohlcv["volume"], errors="coerce").astype(float).rolling(
        window=oi_to_volume_window,
        min_periods=oi_to_volume_window,
    ).sum()
    out["oi_to_volume"] = oi / (volume_roll + 1e-12)
    out["oi_accel"] = out["oi_chg_24"].diff(1)
    return out


def build_interaction_features(
    merged_features: pd.DataFrame,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Build interaction features from already computed funding/OI features."""

    frame = merged_features.copy()
    frame["crowd_long_risk"] = (
        (pd.to_numeric(frame["funding_extreme_pos"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(frame["oi_chg_24"], errors="coerce") > 0)
    ).astype(int)
    frame["crowd_short_risk"] = (
        (pd.to_numeric(frame["funding_extreme_neg"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(frame["oi_chg_24"], errors="coerce") > 0)
    ).astype(int)
    frame["leverage_building"] = (
        (pd.to_numeric(frame["funding_z_30"], errors="coerce").abs() > float(threshold))
        & (pd.to_numeric(frame["oi_z_30"], errors="coerce").abs() > float(threshold))
    ).astype(int)
    return frame


def build_all_futures_features(
    df_ohlcv: pd.DataFrame,
    df_funding: pd.DataFrame,
    df_oi: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build all Stage-9 futures extras features as one frame keyed by timestamp."""

    cfg = config or {}
    funding_cfg = dict((cfg.get("funding") or {}))
    oi_cfg = dict((cfg.get("open_interest") or {}))
    max_fill_gap_bars = int(cfg.get("max_fill_gap_bars", 8))
    timeframe = str(cfg.get("timeframe", "1h"))

    funding_features = build_funding_features(
        df_ohlcv=df_ohlcv,
        df_funding=df_funding,
        settings=funding_cfg,
        timeframe=timeframe,
        max_fill_gap_bars=max_fill_gap_bars,
    )
    oi_features = build_oi_features(
        df_ohlcv=df_ohlcv,
        df_oi=df_oi,
        settings=oi_cfg,
        timeframe=timeframe,
        max_fill_gap_bars=max_fill_gap_bars,
    )

    merged = funding_features.merge(oi_features, on="timestamp", how="left")
    merged = build_interaction_features(merged)
    return merged[["timestamp", *registered_futures_feature_columns()]]


def synthetic_futures_extras(df_ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic synthetic funding/OI aligned to OHLCV timestamps."""

    ts = pd.to_datetime(df_ohlcv["timestamp"], utc=True)
    idx = np.arange(len(ts), dtype=float)

    funding = pd.DataFrame(
        {
            "timestamp": ts,
            "funding_rate": (0.0001 * np.sin(idx / 24.0)) + (0.00005 * np.cos(idx / 12.0)),
        }
    )
    open_interest = pd.DataFrame(
        {
            "timestamp": ts,
            "open_interest": 1_000_000.0 + (idx * 250.0) + (5000.0 * np.sin(idx / 18.0)),
        }
    )
    return funding, open_interest


def _normalize_funding_input(df_ohlcv: pd.DataFrame, df_funding: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if "funding_rate" not in df_funding.columns:
        raise ValueError("funding input must contain funding_rate")
    if "timestamp" in df_funding.columns:
        aligned = df_funding[["timestamp", "funding_rate"]].copy()
        aligned = aligned.rename(columns={"timestamp": "timestamp"})
        aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], utc=True)
        aligned = aligned.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        return df_ohlcv[["timestamp"]].merge(aligned, on="timestamp", how="left")
    if "ts" in df_funding.columns:
        return align_funding_to_ohlcv(ohlcv=df_ohlcv, funding=df_funding[["ts", "funding_rate"]], timeframe=timeframe)
    raise ValueError("funding input must contain either timestamp or ts column")


def _normalize_oi_input(df_ohlcv: pd.DataFrame, df_oi: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if "open_interest" not in df_oi.columns:
        raise ValueError("open-interest input must contain open_interest")
    if "timestamp" in df_oi.columns:
        aligned = df_oi[["timestamp", "open_interest"]].copy()
        aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], utc=True)
        aligned = aligned.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        return df_ohlcv[["timestamp"]].merge(aligned, on="timestamp", how="left")
    if "ts" in df_oi.columns:
        return align_open_interest_to_ohlcv(ohlcv=df_ohlcv, open_interest=df_oi[["ts", "open_interest"]], timeframe=timeframe)
    raise ValueError("open-interest input must contain either timestamp or ts column")


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    window = int(max(1, window))

    def _rank(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or np.isnan(arr).all():
            return np.nan
        return float(np.mean(arr <= arr[-1]))

    return pd.Series(series, dtype=float).rolling(window=window, min_periods=window).apply(_rank, raw=True)
