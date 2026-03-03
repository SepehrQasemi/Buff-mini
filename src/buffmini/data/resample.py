"""Deterministic OHLCV resampling utilities built from base timeframe bars."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from buffmini.data.loader import standardize_ohlcv_frame, validate_ohlcv_frame


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    text = str(timeframe).strip().lower()
    match = re.fullmatch(r"(\d+)([mhdw])", text)
    if not match:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    if unit == "w":
        return pd.Timedelta(weeks=value)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def is_timeframe_multiple(base_timeframe: str, target_timeframe: str) -> bool:
    base = timeframe_to_timedelta(base_timeframe)
    target = timeframe_to_timedelta(target_timeframe)
    return bool(target >= base and target.total_seconds() % base.total_seconds() == 0)


def resample_ohlcv(
    frame: pd.DataFrame,
    target_timeframe: str,
    *,
    base_timeframe: str = "1m",
    partial_last_bucket: bool = False,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Resample OHLCV with exact aggregation and UTC epoch-aligned buckets."""

    if not is_timeframe_multiple(base_timeframe=base_timeframe, target_timeframe=target_timeframe):
        raise ValueError("target_timeframe must be an integer multiple of base_timeframe")
    data = standardize_ohlcv_frame(frame)
    validate_ohlcv_frame(data)
    if data.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    if str(target_timeframe).strip().lower() == str(base_timeframe).strip().lower():
        return data.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()

    freq = _to_pandas_freq(target_timeframe)
    base_delta = timeframe_to_timedelta(base_timeframe)
    target_delta = timeframe_to_timedelta(target_timeframe)
    expected_rows = int(target_delta.total_seconds() // base_delta.total_seconds())

    work = data.rename(columns={timestamp_col: "timestamp"}).set_index("timestamp")
    grouped = work.resample(freq, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    grouped["bar_count"] = work["close"].resample(freq, label="left", closed="left").count()
    grouped = grouped.dropna(subset=["open", "high", "low", "close"])
    if not bool(partial_last_bucket):
        grouped = grouped[grouped["bar_count"] >= expected_rows]
    grouped = grouped.reset_index().rename(columns={"index": "timestamp"})
    out = grouped.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    validate_ohlcv_frame(out)
    return out


def resample_monthly_ohlcv(
    frame: pd.DataFrame,
    *,
    partial_last_bucket: bool = False,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Resample OHLCV into calendar months using UTC month-start boundaries."""

    data = standardize_ohlcv_frame(frame)
    validate_ohlcv_frame(data)
    if data.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    work = data.rename(columns={timestamp_col: "timestamp"}).set_index("timestamp")
    grouped = work.resample("MS", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    grouped = grouped.dropna(subset=["open", "high", "low", "close"])
    if not bool(partial_last_bucket) and not grouped.empty:
        last_source_ts = pd.to_datetime(data["timestamp"], utc=True, errors="coerce").dropna().iloc[-1]
        keep_mask = []
        for bucket_start in pd.to_datetime(grouped.index, utc=True, errors="coerce"):
            bucket_end = bucket_start + pd.offsets.MonthBegin(1)
            keep_mask.append(bool(last_source_ts >= (bucket_end - pd.Timedelta(minutes=1))))
        grouped = grouped.loc[pd.Series(keep_mask, index=grouped.index)]
    out = grouped.reset_index().rename(columns={"index": "timestamp"})
    out = out.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    validate_ohlcv_frame(out)
    return out


def assert_resample_is_causal(
    base_frame: pd.DataFrame,
    resampled_frame: pd.DataFrame,
    *,
    target_timeframe: str,
    base_timeframe: str = "1m",
) -> None:
    """Assert each resampled bar uses only source rows within its closed bucket."""

    base = standardize_ohlcv_frame(base_frame)
    out = standardize_ohlcv_frame(resampled_frame)
    if base.empty or out.empty:
        return

    target_delta = timeframe_to_timedelta(target_timeframe)
    for row in out.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        window = base[(base["timestamp"] >= ts) & (base["timestamp"] < ts + target_delta)]
        if window.empty:
            raise ValueError(f"Causal check failed: empty source window for {ts.isoformat()}")
        open_expected = float(window["open"].iloc[0])
        high_expected = float(window["high"].max())
        low_expected = float(window["low"].min())
        close_expected = float(window["close"].iloc[-1])
        volume_expected = float(window["volume"].sum())
        if (
            abs(float(row.open) - open_expected) > 1e-12
            or abs(float(row.high) - high_expected) > 1e-12
            or abs(float(row.low) - low_expected) > 1e-12
            or abs(float(row.close) - close_expected) > 1e-12
            or abs(float(row.volume) - volume_expected) > 1e-9
        ):
            raise ValueError(f"Causal check failed for bucket {ts.isoformat()}")


def _to_pandas_freq(timeframe: str) -> str:
    text = str(timeframe).strip().lower()
    if text.endswith("m"):
        return f"{int(text[:-1])}min"
    if text.endswith("h"):
        return f"{int(text[:-1])}h"
    if text.endswith("d"):
        return f"{int(text[:-1])}d"
    if text.endswith("w"):
        return f"{int(text[:-1])}W-MON"
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def resample_settings_hash(base_timeframe: str, target_timeframe: str, partial_last_bucket: bool) -> dict[str, Any]:
    return {
        "base_timeframe": str(base_timeframe),
        "target_timeframe": str(target_timeframe),
        "partial_last_bucket": bool(partial_last_bucket),
    }
