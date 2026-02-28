"""Deterministic MTF bar resampling utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buffmini.mtf.spec import timeframe_to_timedelta


def resample_ohlcv(
    base_df: pd.DataFrame,
    target_timeframe: str,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Resample OHLCV to a higher/lower timeframe with explicit bar close timestamps."""

    required = {"open", "high", "low", "close", "volume", timestamp_col}
    missing = required.difference(base_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for resample: {sorted(missing)}")

    data = base_df.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], utc=True, errors="coerce")
    data = data.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(columns=["ts_open", "ts_close", "open", "high", "low", "close", "volume", "timeframe"])

    freq = _pandas_freq(target_timeframe)
    grouped = (
        data.set_index(timestamp_col)
        .resample(rule=freq, label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
    )
    grouped = grouped.dropna(subset=["open", "high", "low", "close"]).reset_index()
    grouped = grouped.rename(columns={timestamp_col: "ts_open"})
    bar_delta = timeframe_to_timedelta(target_timeframe)
    grouped["ts_close"] = pd.to_datetime(grouped["ts_open"], utc=True) + bar_delta
    grouped["timeframe"] = str(target_timeframe)

    ordered_cols = ["ts_open", "ts_close", "open", "high", "low", "close", "volume", "timeframe"]
    return grouped[ordered_cols].reset_index(drop=True)


def validate_resampled_schema(frame: pd.DataFrame) -> None:
    """Validate output schema and temporal consistency."""

    required = {"ts_open", "ts_close", "open", "high", "low", "close", "volume", "timeframe"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing resampled schema columns: {sorted(missing)}")
    if frame.empty:
        return

    ts_open = pd.to_datetime(frame["ts_open"], utc=True, errors="coerce")
    ts_close = pd.to_datetime(frame["ts_close"], utc=True, errors="coerce")
    if ts_open.isna().any() or ts_close.isna().any():
        raise ValueError("resampled timestamps must be valid UTC datetimes")
    if not ts_open.is_monotonic_increasing:
        raise ValueError("resampled ts_open must be monotonic increasing")
    if not ts_close.is_monotonic_increasing:
        raise ValueError("resampled ts_close must be monotonic increasing")
    if not (ts_close > ts_open).all():
        raise ValueError("ts_close must be strictly greater than ts_open")


def _pandas_freq(target_timeframe: str) -> str:
    text = str(target_timeframe).strip().lower()
    if text.endswith("m"):
        return f"{int(text[:-1])}min"
    if text.endswith("h"):
        return f"{int(text[:-1])}h"
    if text.endswith("d"):
        return f"{int(text[:-1])}d"
    raise ValueError(f"Unsupported timeframe for resample: {target_timeframe}")
