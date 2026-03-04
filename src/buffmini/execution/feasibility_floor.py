"""Stage-27.9 feasibility floor math helpers."""

from __future__ import annotations


def calculate_min_risk_pct(
    equity: float,
    stop_distance_pct: float,
    min_notional: float,
    fee_roundtrip_pct: float,
    size_step: float,
) -> float:
    """Minimum risk fraction needed to make a trade feasible.

    Notes:
    - `size_step` is treated as a deterministic notional quantum floor proxy when
      explicit quantity->notional conversion context is unavailable.
    """

    eq = max(float(equity), 1e-12)
    stop = max(float(stop_distance_pct), 0.0)
    fee = max(float(fee_roundtrip_pct), 0.0)
    denom = stop + fee
    if denom <= 0.0:
        return float("inf")
    notional_floor = max(float(min_notional), float(size_step), 0.0)
    return float((notional_floor * denom) / eq)


def calculate_min_equity(
    risk_pct: float,
    stop_distance_pct: float,
    min_notional: float,
) -> float:
    """Minimum equity needed for a trade at fixed risk fraction."""

    rp = max(float(risk_pct), 0.0)
    stop = max(float(stop_distance_pct), 0.0)
    if rp <= 0.0 or stop <= 0.0:
        return float("inf")
    notional_floor = max(float(min_notional), 0.0)
    return float((notional_floor * stop) / rp)

