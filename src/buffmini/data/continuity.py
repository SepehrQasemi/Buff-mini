"""Continuity and missing-candle diagnostics for canonical research mode."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buffmini.data.resample import timeframe_to_timedelta


def continuity_report(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    max_gap_bars: int = 0,
) -> dict[str, Any]:
    """Return deterministic continuity metrics and gap diagnostics."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {
            "rows": 0,
            "gap_count": 0,
            "max_gap_bars": 0,
            "largest_gap_bars": 0,
            "missing_ratio": 0.0,
            "passed": False,
            "passes_strict": False,
            "gaps": [],
        }

    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work.get("timestamp"), utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    if work.empty:
        return {
            "rows": 0,
            "gap_count": 0,
            "max_gap_bars": 0,
            "largest_gap_bars": 0,
            "missing_ratio": 0.0,
            "passed": False,
            "passes_strict": False,
            "gaps": [],
        }

    expected_delta = timeframe_to_timedelta(str(timeframe))
    diffs = work["timestamp"].diff()
    gap_rows: list[dict[str, Any]] = []
    total_missing_bars = 0
    largest_gap = 0
    for idx in range(1, len(work)):
        diff = diffs.iloc[idx]
        if pd.isna(diff) or diff <= expected_delta:
            continue
        bars_missing = int(round(diff / expected_delta)) - 1
        if bars_missing <= 0:
            continue
        total_missing_bars += bars_missing
        largest_gap = max(largest_gap, bars_missing)
        gap_rows.append(
            {
                "start_ts": work["timestamp"].iloc[idx - 1].isoformat(),
                "end_ts": work["timestamp"].iloc[idx].isoformat(),
                "bars_missing": int(bars_missing),
            }
        )

    total_expected_bars = int(len(work) + total_missing_bars)
    missing_ratio = float(total_missing_bars / max(total_expected_bars, 1))
    return {
        "rows": int(len(work)),
        "gap_count": int(len(gap_rows)),
        "max_gap_bars": int(max_gap_bars),
        "largest_gap_bars": int(largest_gap),
        "total_missing_bars": int(total_missing_bars),
        "missing_ratio": float(round(missing_ratio, 12)),
        "passed": bool(largest_gap <= int(max_gap_bars)),
        "passes_strict": bool(largest_gap <= int(max_gap_bars)),
        "gaps": gap_rows,
    }
