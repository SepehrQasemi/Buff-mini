"""Transaction cost helpers."""

from __future__ import annotations

from typing import Literal


def apply_fee(notional: float, round_trip_cost_pct: float) -> float:
    """Return absolute fee cost for a trade notional."""

    return abs(notional) * float(round_trip_cost_pct)


def apply_slippage(price: float, slippage_pct: float, side: Literal["buy", "sell"]) -> float:
    """Adjust execution price for slippage by side."""

    multiplier = 1.0 + float(slippage_pct) if side == "buy" else 1.0 - float(slippage_pct)
    return float(price) * multiplier
