"""Stage-8.2 deterministic cost model v2."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_V2_DEFAULTS = {
    "slippage_bps_base": 0.5,
    "slippage_bps_vol_mult": 2.0,
    "spread_bps": 0.5,
    "delay_bars": 0,
    "vol_proxy": "atr_pct",
    "vol_lookback": 14,
    "max_total_bps_per_side": 10.0,
    "fill_field": "open",
}


def normalize_cost_model_cfg(
    cost_model_cfg: dict[str, Any] | None,
    round_trip_cost_pct: float,
    slippage_pct: float,
) -> dict[str, Any]:
    """Normalize cost model config with backward-compatible defaults."""

    payload = dict(cost_model_cfg or {})
    mode = str(payload.get("mode", "simple")).strip().lower()
    if mode not in {"simple", "v2"}:
        raise ValueError("cost_model.mode must be 'simple' or 'v2'")
    v2_cfg = dict(_V2_DEFAULTS)
    if isinstance(payload.get("v2"), dict):
        v2_cfg.update(payload["v2"])
    v2_cfg["delay_bars"] = int(v2_cfg["delay_bars"])
    v2_cfg["vol_lookback"] = int(v2_cfg["vol_lookback"])
    if v2_cfg["delay_bars"] < 0:
        raise ValueError("cost_model.v2.delay_bars must be >= 0")
    if v2_cfg["vol_lookback"] < 1:
        raise ValueError("cost_model.v2.vol_lookback must be >= 1")
    if str(v2_cfg["vol_proxy"]) != "atr_pct":
        raise ValueError("cost_model.v2.vol_proxy must be 'atr_pct'")
    if float(v2_cfg["max_total_bps_per_side"]) <= 0:
        raise ValueError("cost_model.v2.max_total_bps_per_side must be > 0")
    return {
        "mode": mode,
        "round_trip_cost_pct": float(round_trip_cost_pct),
        "slippage_pct": float(slippage_pct),
        "v2": v2_cfg,
    }


def round_trip_pct_to_one_way_fee_rate(round_trip_cost_pct: float) -> float:
    """Convert round-trip percent (0.1 => 0.1%) into one-way fee rate."""

    pct = float(round_trip_cost_pct)
    if pct < 0:
        raise ValueError("round_trip_cost_pct must be >= 0")
    return (pct / 100.0) / 2.0


def resolve_delay_bars(cost_cfg: dict[str, Any]) -> int:
    """Resolve execution delay bars for the active model."""

    if str(cost_cfg.get("mode", "simple")) != "v2":
        return 0
    return int(cost_cfg.get("v2", {}).get("delay_bars", 0))


def resolve_fill_index(trigger_index: int, delay_bars: int, frame_length: int) -> int:
    """Resolve deterministic fill index with delay and bounds."""

    idx = int(trigger_index) + max(0, int(delay_bars))
    return int(min(max(0, idx), max(0, int(frame_length) - 1)))


def resolve_fill_price_base(
    frame: pd.DataFrame,
    trigger_index: int,
    base_price: float,
    cost_cfg: dict[str, Any],
) -> tuple[float, int]:
    """Resolve execution base price and index after delay."""

    if frame.empty:
        return float(base_price), int(trigger_index)
    delay = resolve_delay_bars(cost_cfg)
    fill_idx = resolve_fill_index(trigger_index=int(trigger_index), delay_bars=delay, frame_length=len(frame))
    if delay <= 0:
        return float(base_price), int(fill_idx)
    field = str(cost_cfg.get("v2", {}).get("fill_field", "open"))
    if field not in frame.columns:
        field = "close" if "close" in frame.columns else field
    fill_value = pd.to_numeric(frame.iloc[fill_idx].get(field, base_price), errors="coerce")
    if pd.isna(fill_value):
        return float(base_price), int(fill_idx)
    return float(fill_value), int(fill_idx)


def one_way_slippage_rate(
    frame: pd.DataFrame,
    bar_index: int,
    cost_cfg: dict[str, Any],
    atr_col: str = "atr_14",
    close_col: str = "close",
) -> float:
    """Return one-way slippage+spread rate for the active model."""

    mode = str(cost_cfg.get("mode", "simple"))
    if mode == "simple":
        return float(cost_cfg.get("slippage_pct", 0.0))
    bps = one_way_cost_breakdown_bps(
        frame=frame,
        bar_index=int(bar_index),
        cost_cfg=cost_cfg,
        atr_col=atr_col,
        close_col=close_col,
    )
    return float(bps["total_bps"] / 10_000.0)


def one_way_cost_breakdown_bps(
    frame: pd.DataFrame,
    bar_index: int,
    cost_cfg: dict[str, Any],
    atr_col: str = "atr_14",
    close_col: str = "close",
) -> dict[str, float]:
    """Return deterministic v2 one-way cost breakdown in bps."""

    v2 = dict(cost_cfg.get("v2", {}))
    spread_bps = max(0.0, float(v2.get("spread_bps", _V2_DEFAULTS["spread_bps"])))
    slip_base = max(0.0, float(v2.get("slippage_bps_base", _V2_DEFAULTS["slippage_bps_base"])))
    slip_mult = max(0.0, float(v2.get("slippage_bps_vol_mult", _V2_DEFAULTS["slippage_bps_vol_mult"])))
    max_total = max(0.0, float(v2.get("max_total_bps_per_side", _V2_DEFAULTS["max_total_bps_per_side"])))
    vol_lookback = max(1, int(v2.get("vol_lookback", _V2_DEFAULTS["vol_lookback"])))

    vol_proxy_bps = atr_pct_proxy_bps(
        frame=frame,
        bar_index=int(bar_index),
        lookback=vol_lookback,
        atr_col=atr_col,
        close_col=close_col,
    )
    dynamic_slip_raw = slip_base + slip_mult * vol_proxy_bps
    total_raw = spread_bps + dynamic_slip_raw
    total_capped = min(max_total, total_raw)

    spread_effective = min(spread_bps, total_capped)
    slip_effective = max(0.0, total_capped - spread_effective)
    return {
        "spread_bps": float(spread_effective),
        "dynamic_slippage_bps": float(slip_effective),
        "vol_proxy_bps": float(vol_proxy_bps),
        "total_bps": float(total_capped),
    }


def atr_pct_proxy_bps(
    frame: pd.DataFrame,
    bar_index: int,
    lookback: int,
    atr_col: str = "atr_14",
    close_col: str = "close",
) -> float:
    """Compute ATR/close proxy in bps using trailing lookback."""

    if frame.empty or atr_col not in frame.columns or close_col not in frame.columns:
        return 0.0
    idx = int(min(max(0, int(bar_index)), len(frame) - 1))
    start = max(0, idx - int(lookback) + 1)
    atr = pd.to_numeric(frame.iloc[start : idx + 1][atr_col], errors="coerce")
    close = pd.to_numeric(frame.iloc[start : idx + 1][close_col], errors="coerce").replace(0.0, np.nan)
    ratio = (atr / close).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return 0.0
    return float(ratio.mean() * 10_000.0)

