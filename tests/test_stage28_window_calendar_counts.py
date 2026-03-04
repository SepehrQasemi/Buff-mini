from __future__ import annotations

import pandas as pd

from buffmini.stage28.window_calendar import expected_window_count, generate_window_calendar


def _timeline_48_months_hourly() -> pd.Series:
    idx = pd.date_range("2022-01-01", "2026-01-01", freq="1h", inclusive="left", tz="UTC")
    return pd.Series(idx)


def test_stage28_window_calendar_count_3m_step_1m() -> None:
    ts = _timeline_48_months_hourly()
    calendar = generate_window_calendar(ts, window_months=3, step_months=1)
    assert int(calendar.shape[0]) == 46
    expected = expected_window_count(
        start_ts=pd.Timestamp(ts.iloc[0]),
        end_ts=pd.Timestamp(ts.iloc[-1]) + pd.Timedelta(hours=1),
        window_months=3,
        step_months=1,
    )
    assert expected == 46


def test_stage28_window_calendar_count_6m_step_1m() -> None:
    ts = _timeline_48_months_hourly()
    calendar = generate_window_calendar(ts, window_months=6, step_months=1)
    assert int(calendar.shape[0]) == 43
    expected = expected_window_count(
        start_ts=pd.Timestamp(ts.iloc[0]),
        end_ts=pd.Timestamp(ts.iloc[-1]) + pd.Timedelta(hours=1),
        window_months=6,
        step_months=1,
    )
    assert expected == 43
