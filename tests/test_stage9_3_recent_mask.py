"""Stage-9.3 recent OI overlay window + masking tests."""

from __future__ import annotations

import pandas as pd

from buffmini.stage9.oi_overlay import (
    OI_DEPENDENT_COLUMNS,
    compute_oi_overlay_window,
    mask_oi_columns,
)


def _ohlcv(rows: int = 48) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "close": range(rows)})


def test_overlay_window_clamps_to_earliest_oi() -> None:
    ohlcv = _ohlcv(rows=96)
    oi = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-03", periods=20, freq="h", tz="UTC"),
            "open_interest": [1000 + i for i in range(20)],
        }
    )
    window = compute_oi_overlay_window(
        df_ohlcv=ohlcv,
        df_oi=oi,
        resolved_end_ts=ohlcv["timestamp"].max(),
        recent_days=30,
    )
    assert window.window_start_ts == pd.Timestamp("2026-01-03T00:00:00Z")
    assert window.note == "CLAMPED"
    assert window.clamped_days >= 0


def test_mask_sets_oi_columns_to_nan_before_window() -> None:
    frame = _ohlcv(rows=30).copy()
    for col in OI_DEPENDENT_COLUMNS:
        frame[col] = 1.0
    start_ts = pd.Timestamp("2026-01-01T12:00:00Z")
    masked, active = mask_oi_columns(frame, ts_col="timestamp", window_start_ts=start_ts, oi_columns=OI_DEPENDENT_COLUMNS)
    before = masked["timestamp"] < start_ts
    after = masked["timestamp"] >= start_ts
    for col in OI_DEPENDENT_COLUMNS:
        assert masked.loc[before, col].isna().all()
    assert active.loc[before].eq(False).all()
    assert active.loc[after].eq(True).all()


def test_mask_with_no_oi_window_disables_all() -> None:
    frame = _ohlcv(rows=10).copy()
    for col in OI_DEPENDENT_COLUMNS:
        frame[col] = 3.14
    masked, active = mask_oi_columns(frame, ts_col="timestamp", window_start_ts=None, oi_columns=OI_DEPENDENT_COLUMNS)
    for col in OI_DEPENDENT_COLUMNS:
        assert masked[col].isna().all()
    assert active.eq(False).all()

