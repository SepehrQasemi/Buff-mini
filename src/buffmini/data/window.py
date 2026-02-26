"""Time-window slicing helpers for evaluation runs."""

from __future__ import annotations

import pandas as pd


def slice_last_n_months(
    frame: pd.DataFrame,
    window_months: int,
    end_mode: str = "latest",
) -> tuple[pd.DataFrame, str]:
    """Slice frame to last N months and return sliced frame plus date range string."""

    if frame.empty:
        return frame.copy(), "n/a"
    if int(window_months) < 1:
        raise ValueError("window_months must be >= 1")
    if end_mode != "latest":
        raise ValueError("Only end_mode='latest' is supported in Stage-0.6")

    data = frame.copy().sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(data["timestamp"], utc=True)
    end_ts = timestamps.max()
    start_ts = end_ts - pd.DateOffset(months=int(window_months))

    sliced = data.loc[timestamps >= start_ts].reset_index(drop=True)
    if sliced.empty:
        sliced = data.copy()

    sliced_ts = pd.to_datetime(sliced["timestamp"], utc=True)
    date_range = f"{sliced_ts.iloc[0].isoformat()}..{sliced_ts.iloc[-1].isoformat()}"
    return sliced, date_range
