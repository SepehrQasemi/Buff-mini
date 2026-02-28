"""Transaction cost helpers."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from buffmini.validation.cost_model_v2 import (
    normalize_cost_model_cfg,
    one_way_cost_breakdown_bps,
    one_way_slippage_rate,
    resolve_fill_index,
    resolve_fill_price_base,
)


def round_trip_pct_to_one_way_fee_rate(round_trip_cost_pct: float) -> float:
    """Convert round-trip percent cost (e.g., 0.1 for 0.1%) to one-way fee rate."""

    pct = float(round_trip_cost_pct)
    if pct < 0:
        raise ValueError("round_trip_cost_pct must be >= 0")
    return (pct / 100.0) / 2.0


def apply_fee(notional: float, fee_rate: float) -> float:
    """Return absolute fee cost for a trade notional and one-way fee rate."""

    return abs(notional) * float(fee_rate)


def apply_slippage(price: float, slippage_pct: float, side: Literal["buy", "sell"]) -> float:
    """Adjust execution price for slippage by side."""

    multiplier = 1.0 + float(slippage_pct) if side == "buy" else 1.0 - float(slippage_pct)
    return float(price) * multiplier


def normalized_cost_cfg(
    round_trip_cost_pct: float,
    slippage_pct: float,
    cost_model_cfg: dict | None,
) -> dict:
    """Normalize cost model config for engine usage."""

    return normalize_cost_model_cfg(
        cost_model_cfg=cost_model_cfg,
        round_trip_cost_pct=float(round_trip_cost_pct),
        slippage_pct=float(slippage_pct),
    )


def one_way_slippage_for_bar(
    frame: pd.DataFrame,
    bar_index: int,
    cost_cfg: dict,
    atr_col: str = "atr_14",
    close_col: str = "close",
) -> float:
    """Resolve one-way slippage/spread execution rate for one bar."""

    return one_way_slippage_rate(
        frame=frame,
        bar_index=int(bar_index),
        cost_cfg=cost_cfg,
        atr_col=atr_col,
        close_col=close_col,
    )


def fill_price_and_index(
    frame: pd.DataFrame,
    trigger_index: int,
    base_price: float,
    cost_cfg: dict,
) -> tuple[float, int]:
    """Resolve deterministic fill base price and index with optional delay."""

    return resolve_fill_price_base(
        frame=frame,
        trigger_index=int(trigger_index),
        base_price=float(base_price),
        cost_cfg=cost_cfg,
    )


def cost_breakdown_bps(
    frame: pd.DataFrame,
    bar_index: int,
    cost_cfg: dict,
    atr_col: str = "atr_14",
    close_col: str = "close",
) -> dict[str, float]:
    """Return one-way cost breakdown in bps (v2 mode) for reporting/tests."""

    return one_way_cost_breakdown_bps(
        frame=frame,
        bar_index=int(bar_index),
        cost_cfg=cost_cfg,
        atr_col=atr_col,
        close_col=close_col,
    )


def delayed_index(trigger_index: int, delay_bars: int, frame_length: int) -> int:
    """Expose delayed fill index helper for deterministic tests."""

    return resolve_fill_index(
        trigger_index=int(trigger_index),
        delay_bars=int(delay_bars),
        frame_length=int(frame_length),
    )

