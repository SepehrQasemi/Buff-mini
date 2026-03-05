"""Stage-34 label generation (primary + auxiliary)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.stage26.coverage import timeframe_seconds


def build_labels(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    horizons_hours: list[int] | None = None,
) -> pd.DataFrame:
    horizons = [int(v) for v in (horizons_hours or [24, 72]) if int(v) > 0]
    work = frame.copy().sort_values("timestamp").reset_index(drop=True)
    close = pd.to_numeric(work.get("close"), errors="coerce")
    high = pd.to_numeric(work.get("high"), errors="coerce")
    low = pd.to_numeric(work.get("low"), errors="coerce")
    atr_pct = pd.to_numeric(work.get("atr_pct"), errors="coerce")
    if atr_pct.isna().all():
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_pct = tr.rolling(14, min_periods=2).mean() / close.replace(0.0, np.nan)
    atr_pct = atr_pct.ffill().fillna(0.0).clip(lower=1e-6)

    out = pd.DataFrame({"timestamp": pd.to_datetime(work.get("timestamp"), utc=True, errors="coerce")})
    primary_col = None
    aux_col = None
    for horizon_h in horizons:
        bars = max(1, int(round((float(horizon_h) * 3600.0) / float(timeframe_seconds(timeframe)))))
        future_close = close.shift(-bars)
        final_ret = future_close / close.replace(0.0, np.nan) - 1.0
        f_high = _future_rolling_extreme(high, bars=bars, mode="max")
        f_low = _future_rolling_extreme(low, bars=bars, mode="min")
        max_ret = f_high / close.replace(0.0, np.nan) - 1.0
        min_ret = f_low / close.replace(0.0, np.nan) - 1.0

        tp = (atr_pct * 1.5 + 0.001).clip(lower=0.001)
        sl = (atr_pct * 1.0 + 0.001).clip(lower=0.001)
        triple = np.where(
            (max_ret >= tp) & (min_ret <= -sl),
            np.sign(final_ret.fillna(0.0)),
            np.where(
                max_ret >= tp,
                1.0,
                np.where(min_ret <= -sl, -1.0, np.sign(final_ret.fillna(0.0))),
            ),
        )
        triple = pd.Series(triple, index=out.index, dtype=float).fillna(0.0)
        # keep neutral when tiny move
        triple = np.where(final_ret.abs() < 0.0005, 0.0, triple)
        triple = pd.Series(triple, index=out.index, dtype=float).clip(-1.0, 1.0)
        aux = pd.to_numeric(min_ret, errors="coerce").fillna(0.0)

        tri_col = f"label_triple_{int(horizon_h)}h"
        aux_name = f"label_fwd_adverse_{int(horizon_h)}h"
        out[tri_col] = triple.astype(int)
        out[aux_name] = aux.astype(float)
        if primary_col is None:
            primary_col = tri_col
            aux_col = aux_name
    out["label_primary"] = out[primary_col].astype(int) if primary_col is not None else 0
    out["label_auxiliary"] = pd.to_numeric(out[aux_col], errors="coerce").fillna(0.0) if aux_col is not None else 0.0
    return out


def _future_rolling_extreme(series: pd.Series, *, bars: int, mode: str) -> pd.Series:
    rev = pd.to_numeric(series, errors="coerce").iloc[::-1]
    shifted = rev.shift(1)
    if mode == "max":
        rolled = shifted.rolling(int(bars), min_periods=int(bars)).max()
    elif mode == "min":
        rolled = shifted.rolling(int(bars), min_periods=int(bars)).min()
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return rolled.iloc[::-1]


def label_columns() -> list[str]:
    return ["label_primary", "label_auxiliary"]
