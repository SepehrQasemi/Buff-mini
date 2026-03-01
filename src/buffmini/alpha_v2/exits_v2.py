"""Stage-17 exit engine v2 helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


ExitMode = Literal["fixed_atr", "atr_trailing", "time_progress", "partial_runner", "mae_mfe_tighten"]


@dataclass(frozen=True)
class ExitConfigV2:
    mode: ExitMode = "fixed_atr"
    stop_atr_multiple: float = 1.5
    take_profit_atr_multiple: float = 3.0
    trailing_atr_k: float = 1.5
    max_hold_bars: int = 24
    progress_window: int = 8
    min_progress_atr: float = 0.15
    partial_fraction: float = 0.5


def trailing_stop_series(
    *,
    close: pd.Series,
    atr: pd.Series,
    side: int,
    trailing_k: float,
) -> pd.Series:
    """Deterministic volatility-adjusted trailing stop path."""

    c = pd.to_numeric(close, errors="coerce").fillna(method="ffill").fillna(0.0)
    a = pd.to_numeric(atr, errors="coerce").replace(0.0, np.nan).fillna(method="ffill").fillna(1.0)
    if side >= 0:
        peak = c.cummax()
        stop = peak - float(trailing_k) * a
    else:
        trough = c.cummin()
        stop = trough + float(trailing_k) * a
    return stop.astype(float)


def progress_stop_flags(
    *,
    close: pd.Series,
    atr: pd.Series,
    entry_index: int,
    side: int,
    progress_window: int,
    min_progress_atr: float,
) -> pd.Series:
    """Exit flag when no minimum progress after N bars."""

    c = pd.to_numeric(close, errors="coerce").fillna(method="ffill").fillna(0.0)
    a = pd.to_numeric(atr, errors="coerce").replace(0.0, np.nan).fillna(method="ffill").fillna(1.0)
    flags = np.zeros(len(c), dtype=bool)
    if entry_index < 0 or entry_index >= len(c):
        return pd.Series(flags, index=c.index, dtype=bool)
    entry_price = float(c.iloc[entry_index])
    start = int(entry_index + progress_window)
    if start < len(c):
        window_price = float(c.iloc[start])
        progress = (window_price - entry_price) if side >= 0 else (entry_price - window_price)
        if progress < float(min_progress_atr) * float(a.iloc[entry_index]):
            flags[start:] = True
    return pd.Series(flags, index=c.index, dtype=bool)


def mae_mfe_tightening_multiplier(mae: float, mfe: float) -> float:
    """Deterministic tightening factor based on MAE/MFE."""

    if mfe <= 0:
        return 1.0
    ratio = float(max(0.0, mae) / max(1e-12, mfe))
    return float(np.clip(1.0 - 0.3 * ratio, 0.5, 1.0))

