"""Deterministic dataset indexing for Stage-30 self-supervised training."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buffmini.utils.hashing import stable_hash


def build_dataset_index(
    *,
    frames_by_symbol: dict[str, pd.DataFrame],
    timeframe: str,
    window: int,
    stride: int,
) -> pd.DataFrame:
    """Build deterministic rolling window index from per-symbol frames."""

    if int(window) <= 1:
        raise ValueError("window must be > 1")
    if int(stride) <= 0:
        raise ValueError("stride must be > 0")

    rows: list[dict[str, Any]] = []
    for symbol in sorted(frames_by_symbol.keys()):
        frame = frames_by_symbol[str(symbol)].reset_index(drop=True).copy()
        ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
        if ts.isna().all():
            continue
        n = int(frame.shape[0])
        start = 0
        while start + int(window) <= n:
            end = int(start + window)
            start_ts = ts.iloc[start]
            end_ts = ts.iloc[end - 1]
            if pd.isna(start_ts) or pd.isna(end_ts):
                start += int(stride)
                continue
            rows.append(
                {
                    "symbol": str(symbol),
                    "timeframe": str(timeframe),
                    "row_start": int(start),
                    "row_end_exclusive": int(end),
                    "start_ts": pd.Timestamp(start_ts).tz_convert("UTC").isoformat(),
                    "end_ts": pd.Timestamp(end_ts).tz_convert("UTC").isoformat(),
                    "window_len": int(window),
                    "stride": int(stride),
                }
            )
            start += int(stride)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["symbol", "timeframe", "row_start", "row_end_exclusive"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    out["window_id"] = [
        stable_hash(
            {
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "row_start": int(row["row_start"]),
                "row_end_exclusive": int(row["row_end_exclusive"]),
            },
            length=16,
        )
        for row in out.to_dict(orient="records")
    ]
    return out


def infer_resolved_end_ts(frames_by_symbol: dict[str, pd.DataFrame]) -> str | None:
    """Return max timestamp across all frames, in UTC ISO format."""

    max_ts: pd.Timestamp | None = None
    for frame in frames_by_symbol.values():
        ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
        if ts.empty:
            continue
        tail = pd.Timestamp(ts.max()).tz_convert("UTC")
        if max_ts is None or tail > max_ts:
            max_ts = tail
    return max_ts.isoformat() if max_ts is not None else None

