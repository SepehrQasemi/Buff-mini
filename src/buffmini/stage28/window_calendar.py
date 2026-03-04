"""Deterministic rolling window calendar helpers for Stage-28."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


WINDOW_CALENDAR_COLUMNS: tuple[str, ...] = (
    "window_index",
    "window_start",
    "window_end",
    "row_count",
)


def expected_window_count(
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    window_months: int,
    step_months: int,
) -> int:
    """Return expected rolling window count for month-based windows.

    The right boundary is treated as exclusive: a window [start, end) is valid
    when `end <= end_ts`.
    """

    if int(window_months) <= 0 or int(step_months) <= 0:
        raise ValueError("window_months and step_months must be positive integers")

    start = _to_utc_ts(start_ts)
    end = _to_utc_ts(end_ts)
    if start >= end:
        return 0

    # `end_ts` is typically the timestamp of the last closed bar; treat the
    # right boundary as end-exclusive by adding one inferred bar step.
    end_exclusive = end + pd.Timedelta(minutes=1)

    count = 0
    cursor = start
    while True:
        right = cursor + pd.DateOffset(months=int(window_months))
        if right > end_exclusive:
            break
        count += 1
        cursor = cursor + pd.DateOffset(months=int(step_months))
    return int(count)


def generate_window_calendar(
    timestamps: Iterable[object] | pd.Series,
    *,
    window_months: int,
    step_months: int,
) -> pd.DataFrame:
    """Generate deterministic rolling windows with per-window row counts."""

    if int(window_months) <= 0 or int(step_months) <= 0:
        raise ValueError("window_months and step_months must be positive integers")

    ts = _normalize_timestamps(timestamps)
    if ts.empty:
        return pd.DataFrame(columns=list(WINDOW_CALENDAR_COLUMNS))

    start = pd.Timestamp(ts.iloc[0]).tz_convert("UTC")
    end = pd.Timestamp(ts.iloc[-1]).tz_convert("UTC")
    end_exclusive = end + _infer_step(ts)

    rows: list[dict[str, object]] = []
    cursor = start
    window_index = 0
    while True:
        right = cursor + pd.DateOffset(months=int(window_months))
        if right > end_exclusive:
            break
        mask = (ts >= cursor) & (ts < right)
        rows.append(
            {
                "window_index": int(window_index),
                "window_start": cursor.isoformat(),
                "window_end": right.isoformat(),
                "row_count": int(mask.sum()),
            }
        )
        window_index += 1
        cursor = cursor + pd.DateOffset(months=int(step_months))
    return pd.DataFrame(rows, columns=list(WINDOW_CALENDAR_COLUMNS))


def _normalize_timestamps(timestamps: Iterable[object] | pd.Series) -> pd.Series:
    ts = pd.to_datetime(list(timestamps), utc=True, errors="coerce")
    if isinstance(ts, pd.DatetimeIndex):
        series = pd.Series(ts)
    else:
        series = pd.Series(ts)
    series = series.dropna().sort_values().drop_duplicates().reset_index(drop=True)
    return series


def _to_utc_ts(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="raise")
    return pd.Timestamp(ts).tz_convert("UTC")


def _infer_step(ts: pd.Series) -> pd.Timedelta:
    if len(ts) < 2:
        return pd.Timedelta(minutes=1)
    deltas = ts.diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return pd.Timedelta(minutes=1)
    return pd.Timedelta(deltas.median())
