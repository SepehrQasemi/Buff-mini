from __future__ import annotations

import pandas as pd

from buffmini.stage28.window_calendar import generate_window_calendar


def test_stage28_window_calendar_monotonic_and_month_step() -> None:
    ts = pd.Series(pd.date_range("2022-01-01", "2026-01-01", freq="1h", inclusive="left", tz="UTC"))
    cal = generate_window_calendar(ts, window_months=3, step_months=1)
    assert not cal.empty

    starts = pd.to_datetime(cal["window_start"], utc=True, errors="coerce")
    ends = pd.to_datetime(cal["window_end"], utc=True, errors="coerce")

    assert bool(starts.is_monotonic_increasing)
    assert bool(ends.is_monotonic_increasing)
    assert bool((ends > starts).all())
    assert bool((cal["row_count"] > 0).all())

    for idx in range(1, len(starts)):
        assert starts.iloc[idx] == starts.iloc[idx - 1] + pd.DateOffset(months=1)
