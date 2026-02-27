"""Rolling forward window helpers for Stage-2.8 probabilistic evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


BAR_DELTA = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class RollingWindowSpec:
    """Resolved rolling window inside the reserved forward tail."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    truncated: bool
    enough_data: bool
    bar_count: int
    note: str


@dataclass(frozen=True)
class ReservedTailRange:
    """Reserved tail boundaries used to build rolling windows."""

    start: pd.Timestamp
    end: pd.Timestamp


def build_rolling_windows(
    start_ts: pd.Timestamp | str,
    end_ts: pd.Timestamp | str,
    window_days: int,
    stride_days: int,
    reserve_tail_days: int,
) -> tuple[ReservedTailRange, list[RollingWindowSpec]]:
    """Build overlapping rolling windows constrained to a reserved tail.

    The returned windows overlap by design when ``stride_days < window_days``. The
    caller is expected to pass ``start_ts`` as the earliest permissible timestamp
    after holdout end so windows remain strictly out-of-sample.
    """

    if int(window_days) < 1:
        raise ValueError("window_days must be >= 1")
    if int(stride_days) < 1:
        raise ValueError("stride_days must be >= 1")
    if int(reserve_tail_days) < 0:
        raise ValueError("reserve_tail_days must be >= 0")

    earliest_allowed = _ensure_utc(start_ts)
    latest_available = _ensure_utc(end_ts)
    if earliest_allowed > latest_available:
        return ReservedTailRange(start=earliest_allowed, end=latest_available), []

    if int(reserve_tail_days) == 0:
        tail_start = earliest_allowed
    else:
        tail_start = latest_available - pd.Timedelta(days=int(reserve_tail_days)) + BAR_DELTA
        if tail_start < earliest_allowed:
            tail_start = earliest_allowed
    reserved_tail = ReservedTailRange(start=tail_start, end=latest_available)
    if reserved_tail.start > reserved_tail.end:
        return reserved_tail, []

    window_delta = pd.Timedelta(days=int(window_days))
    stride_delta = pd.Timedelta(days=int(stride_days))
    windows: list[RollingWindowSpec] = []
    current_start = reserved_tail.start
    index = 1
    while current_start <= reserved_tail.end:
        expected_end = current_start + window_delta - BAR_DELTA
        actual_end = min(expected_end, reserved_tail.end)
        truncated = bool(actual_end < expected_end)
        bar_count = int(((actual_end - current_start) / BAR_DELTA) + 1) if actual_end >= current_start else 0
        enough_data = bool(bar_count >= 2)
        note = "" if not truncated else "Window truncated at local data end within reserved tail."
        windows.append(
            RollingWindowSpec(
                name=f"Window_{index:02d}",
                start=current_start,
                end=actual_end,
                truncated=truncated,
                enough_data=enough_data,
                bar_count=bar_count,
                note=note,
            )
        )
        current_start = current_start + stride_delta
        index += 1
    return reserved_tail, windows


def _ensure_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")
