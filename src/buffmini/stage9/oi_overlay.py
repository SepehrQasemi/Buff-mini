"""Stage-9.3 recent OI overlay windowing and masking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

OI_DEPENDENT_COLUMNS: tuple[str, ...] = (
    "oi",
    "oi_chg_1",
    "oi_chg_24",
    "oi_z_30",
    "oi_to_volume",
    "oi_accel",
    "crowd_long_risk",
    "crowd_short_risk",
    "leverage_building",
)


@dataclass(frozen=True)
class OIOverlayWindow:
    """Resolved recent OI overlay window."""

    window_start_ts: pd.Timestamp | None
    window_end_ts: pd.Timestamp
    clamped_days: int
    note: str
    earliest_oi_ts: pd.Timestamp | None


def compute_oi_overlay_window(
    df_ohlcv: pd.DataFrame,
    df_oi: pd.DataFrame,
    resolved_end_ts: str | pd.Timestamp | None,
    recent_days: int,
) -> OIOverlayWindow:
    """Compute clamped recent OI overlay window."""

    if "timestamp" not in df_ohlcv.columns:
        raise ValueError("df_ohlcv must include timestamp")
    timestamps = pd.to_datetime(df_ohlcv["timestamp"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        raise ValueError("df_ohlcv timestamp column is empty")

    if resolved_end_ts is None or str(resolved_end_ts).strip() == "":
        window_end_ts = timestamps.max()
    else:
        end_ts = pd.Timestamp(resolved_end_ts)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        window_end_ts = min(end_ts, timestamps.max())

    requested_start = window_end_ts - pd.Timedelta(days=int(recent_days))

    oi_ts = _extract_oi_timestamps(df_oi)
    if oi_ts.empty:
        return OIOverlayWindow(
            window_start_ts=None,
            window_end_ts=window_end_ts,
            clamped_days=0,
            note="NO_OI_DATA",
            earliest_oi_ts=None,
        )

    earliest_oi_ts = oi_ts.min()
    window_start_ts = max(requested_start, earliest_oi_ts)
    clamped_days = int(max(0, (window_end_ts - window_start_ts).total_seconds() // 86400))
    note = "CLAMPED" if earliest_oi_ts > requested_start else "OK"
    return OIOverlayWindow(
        window_start_ts=window_start_ts,
        window_end_ts=window_end_ts,
        clamped_days=clamped_days,
        note=note,
        earliest_oi_ts=earliest_oi_ts,
    )


def mask_oi_columns(
    features_df: pd.DataFrame,
    ts_col: str,
    window_start_ts: pd.Timestamp | None,
    oi_columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Mask OI-dependent columns to NaN before overlay window start."""

    if ts_col not in features_df.columns:
        raise ValueError(f"{ts_col} not found in features_df")

    frame = features_df.copy()
    timestamps = pd.to_datetime(frame[ts_col], utc=True, errors="coerce")
    selected_cols = [col for col in (oi_columns or OI_DEPENDENT_COLUMNS) if col in frame.columns]

    if window_start_ts is None:
        for col in selected_cols:
            frame[col] = float("nan")
        oi_active = pd.Series(False, index=frame.index, dtype=bool)
        return frame, oi_active

    start_ts = pd.Timestamp(window_start_ts)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")

    pre_mask = timestamps < start_ts
    for col in selected_cols:
        frame.loc[pre_mask, col] = float("nan")

    required_base = [col for col in ["oi"] if col in frame.columns]
    if not required_base:
        oi_active = pd.Series(False, index=frame.index, dtype=bool)
    else:
        base_non_nan = frame[required_base].notna().all(axis=1)
        oi_active = (~pre_mask) & base_non_nan
    return frame, oi_active.astype(bool)


def _extract_oi_timestamps(df_oi: pd.DataFrame) -> pd.Series:
    if df_oi.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    ts_col = "ts" if "ts" in df_oi.columns else "timestamp" if "timestamp" in df_oi.columns else None
    if ts_col is None:
        return pd.Series(dtype="datetime64[ns, UTC]")

    ts = pd.to_datetime(df_oi[ts_col], utc=True, errors="coerce")
    if "open_interest" in df_oi.columns:
        oi_values = pd.to_numeric(df_oi["open_interest"], errors="coerce")
        ts = ts[oi_values.notna()]
    return ts.dropna().sort_values().reset_index(drop=True)


def overlay_metadata_dict(window: OIOverlayWindow, oi_active: pd.Series, total_rows: int) -> dict[str, Any]:
    """Build serializable overlay metadata payload."""

    active_count = int(pd.Series(oi_active, dtype=bool).sum())
    total = int(total_rows)
    active_percent = float((active_count / total) * 100.0) if total > 0 else 0.0
    return {
        "oi_window_start_ts": window.window_start_ts.isoformat() if window.window_start_ts is not None else None,
        "oi_window_end_ts": window.window_end_ts.isoformat() if window.window_end_ts is not None else None,
        "oi_clamped_days": int(window.clamped_days),
        "oi_window_note": str(window.note),
        "oi_earliest_ts": window.earliest_oi_ts.isoformat() if window.earliest_oi_ts is not None else None,
        "oi_active_percent": float(active_percent),
        "oi_active_rows": active_count,
        "total_rows": total,
    }

