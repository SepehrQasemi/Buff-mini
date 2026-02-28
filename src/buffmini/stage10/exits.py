"""Stage-10.3 exit library utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


EXIT_MODE_ALIASES: dict[str, str] = {
    "fixed_atr": "fixed_atr",
    "atr_trailing": "trailing_atr",
    "breakeven_1r": "breakeven_1r",
    "partial_tp": "partial_then_trail",
    "regime_flip_exit": "fixed_atr",
}


def normalize_exit_mode(mode: str) -> str:
    """Map Stage-10 exit names onto backtest-engine compatible modes."""

    key = str(mode)
    if key not in EXIT_MODE_ALIASES:
        raise ValueError(f"Unknown exit mode: {mode}")
    return EXIT_MODE_ALIASES[key]


def decide_exit_reason(
    stop_hit: bool,
    take_profit_hit: bool,
    time_stop_hit: bool = False,
) -> str:
    """Return deterministic exit reason with stop-first priority."""

    if bool(stop_hit):
        return "stop_loss"
    if bool(take_profit_hit):
        return "take_profit"
    if bool(time_stop_hit):
        return "time_stop"
    return ""


def update_trailing_stop(
    side: str,
    previous_stop: float,
    highest_since_entry: float,
    lowest_since_entry: float,
    atr_value: float,
    trailing_k: float,
) -> float:
    """Update trailing stop with monotonic behavior by side."""

    atr = max(0.0, float(atr_value))
    k = max(0.0, float(trailing_k))
    prev = float(previous_stop)
    if str(side) == "long":
        candidate = float(highest_since_entry) - (k * atr)
        return float(max(prev, candidate))
    if str(side) == "short":
        candidate = float(lowest_since_entry) + (k * atr)
        return float(min(prev, candidate))
    raise ValueError("side must be 'long' or 'short'")


def trailing_stop_path(
    side: str,
    entry_stop: float,
    highs: Iterable[float],
    lows: Iterable[float],
    atr_values: Iterable[float],
    trailing_k: float,
) -> pd.Series:
    """Build deterministic trailing-stop path for tests/reporting."""

    highs_arr = np.asarray(list(highs), dtype=float)
    lows_arr = np.asarray(list(lows), dtype=float)
    atr_arr = np.asarray(list(atr_values), dtype=float)
    if highs_arr.size != lows_arr.size or highs_arr.size != atr_arr.size:
        raise ValueError("highs, lows, atr_values must have equal lengths")
    if highs_arr.size == 0:
        return pd.Series(dtype=float)

    stops = np.empty(highs_arr.size, dtype=float)
    stop = float(entry_stop)
    running_high = highs_arr[0]
    running_low = lows_arr[0]
    for idx in range(highs_arr.size):
        running_high = max(running_high, highs_arr[idx])
        running_low = min(running_low, lows_arr[idx])
        stop = update_trailing_stop(
            side=side,
            previous_stop=stop,
            highest_since_entry=running_high,
            lowest_since_entry=running_low,
            atr_value=atr_arr[idx],
            trailing_k=trailing_k,
        )
        stops[idx] = stop
    return pd.Series(stops)


def apply_breakeven_after_1r(
    side: str,
    entry_price: float,
    current_stop: float,
    one_r_reached: bool,
) -> float:
    """Move stop to entry after +1R trigger."""

    if not bool(one_r_reached):
        return float(current_stop)
    if str(side) == "long":
        return float(max(float(current_stop), float(entry_price)))
    if str(side) == "short":
        return float(min(float(current_stop), float(entry_price)))
    raise ValueError("side must be 'long' or 'short'")


def partial_take_profit(
    side: str,
    entry_price: float,
    target_price: float,
    qty_total: float,
    partial_fraction: float,
) -> dict[str, float]:
    """Compute deterministic partial-take-profit accounting."""

    qty = max(0.0, float(qty_total))
    fraction = min(1.0, max(0.0, float(partial_fraction)))
    partial_qty = qty * fraction
    remaining_qty = qty - partial_qty

    direction = 1.0 if str(side) == "long" else -1.0 if str(side) == "short" else None
    if direction is None:
        raise ValueError("side must be 'long' or 'short'")
    realized_pnl = (float(target_price) - float(entry_price)) * partial_qty * direction

    return {
        "partial_qty": float(partial_qty),
        "remaining_qty": float(remaining_qty),
        "realized_pnl": float(realized_pnl),
    }


def should_regime_flip_exit(
    entry_regime: str,
    current_regime: str,
    exit_regimes: Iterable[str] = ("CHOP",),
) -> bool:
    """Return True when regime-flip exit condition is met."""

    entry = str(entry_regime)
    current = str(current_regime)
    exits = {str(item) for item in exit_regimes}
    return current in exits and current != entry
