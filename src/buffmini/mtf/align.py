"""Causal multi-timeframe alignment helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.mtf.spec import MtfLayerSpec, timeframe_to_timedelta


def join_mtf_layer(
    base_df: pd.DataFrame,
    layer_df: pd.DataFrame,
    layer_spec: MtfLayerSpec,
    base_ts_col: str = "timestamp",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Causally join one layer onto base timeframe rows via backward asof."""

    if base_ts_col not in base_df.columns:
        raise ValueError(f"Missing base timestamp column: {base_ts_col}")
    if "ts_close" not in layer_df.columns:
        raise ValueError("layer_df must include ts_close")

    base = base_df.copy()
    base[base_ts_col] = pd.to_datetime(base[base_ts_col], utc=True, errors="coerce")
    base = base.dropna(subset=[base_ts_col]).sort_values(base_ts_col).reset_index(drop=True)

    layer = layer_df.copy()
    layer["ts_close"] = pd.to_datetime(layer["ts_close"], utc=True, errors="coerce")
    layer = layer.dropna(subset=["ts_close"]).sort_values("ts_close").reset_index(drop=True)

    prefixed = _prefix_layer_columns(layer, layer_spec.name)
    close_col = f"{layer_spec.name}__ts_close"
    tolerance = layer_spec.tolerance_delta(base_timeframe="1h")

    joined = pd.merge_asof(
        left=base,
        right=prefixed,
        left_on=base_ts_col,
        right_on=close_col,
        direction="backward",
        tolerance=tolerance,
    )

    if close_col in joined.columns:
        matched = joined[close_col].notna()
        if matched.any():
            valid = pd.to_datetime(joined.loc[matched, close_col], utc=True) <= pd.to_datetime(
                joined.loc[matched, base_ts_col], utc=True
            )
            if not bool(valid.all()):
                raise ValueError("Causality violation: joined layer close exceeds base timestamp")

    stats = _alignment_stats(
        joined=joined,
        base_ts_col=base_ts_col,
        layer_close_col=close_col,
        layer_timeframe=layer_spec.timeframe,
    )
    return joined, stats


def assert_causal_alignment(joined: pd.DataFrame, base_ts_col: str, layer_close_col: str) -> None:
    """Assert no row contains future layer timestamps."""

    if layer_close_col not in joined.columns:
        return
    base_ts = pd.to_datetime(joined[base_ts_col], utc=True, errors="coerce")
    layer_close = pd.to_datetime(joined[layer_close_col], utc=True, errors="coerce")
    mask = layer_close.notna() & base_ts.notna()
    if not mask.any():
        return
    if not bool((layer_close[mask] <= base_ts[mask]).all()):
        raise ValueError("Found future leakage in MTF alignment")


def _prefix_layer_columns(layer: pd.DataFrame, layer_name: str) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for col in layer.columns:
        renamed[col] = f"{layer_name}__{col}"
    return layer.rename(columns=renamed)


def _alignment_stats(
    joined: pd.DataFrame,
    base_ts_col: str,
    layer_close_col: str,
    layer_timeframe: str,
) -> dict[str, Any]:
    total = int(len(joined))
    if total <= 0 or layer_close_col not in joined.columns:
        return {
            "rows": total,
            "matched_rows": 0,
            "unmatched_rows": total,
            "unmatched_pct": 100.0 if total > 0 else 0.0,
            "max_lookback_bars": 0.0,
        }
    base_ts = pd.to_datetime(joined[base_ts_col], utc=True, errors="coerce")
    layer_close = pd.to_datetime(joined[layer_close_col], utc=True, errors="coerce")
    matched_mask = base_ts.notna() & layer_close.notna()
    matched = int(matched_mask.sum())
    unmatched = int(total - matched)
    if matched == 0:
        max_lookback_bars = 0.0
    else:
        lookback_seconds = (base_ts[matched_mask] - layer_close[matched_mask]).dt.total_seconds().to_numpy(dtype=float)
        layer_seconds = max(1.0, timeframe_to_timedelta(layer_timeframe).total_seconds())
        max_lookback_bars = float(np.max(lookback_seconds) / layer_seconds) if lookback_seconds.size else 0.0
    return {
        "rows": total,
        "matched_rows": matched,
        "unmatched_rows": unmatched,
        "unmatched_pct": float(unmatched / total * 100.0) if total > 0 else 0.0,
        "max_lookback_bars": max_lookback_bars,
    }

