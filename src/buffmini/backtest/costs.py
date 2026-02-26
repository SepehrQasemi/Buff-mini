"""Transaction cost helpers."""

from __future__ import annotations

from typing import Literal


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
