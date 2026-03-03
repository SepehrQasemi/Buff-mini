"""Execution feasibility boundary math for Stage-27 diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def min_required_risk_pct(
    *,
    equity: float,
    min_notional: float,
    stop_dist_pct: float,
    cost_rt_pct: float,
    max_notional_pct: float,
) -> float:
    """Minimum risk percentage required for a trade to satisfy min notional and caps."""

    eq = max(float(equity), 1e-12)
    min_notional_val = max(float(min_notional), 0.0)
    denom = float(stop_dist_pct) + float(cost_rt_pct)
    if denom <= 0.0:
        return float("inf")
    cap = max(float(max_notional_pct), 0.0) * eq
    if cap <= 0.0 or min_notional_val > cap + 1e-12:
        return float("inf")
    return float((min_notional_val * denom) / eq)


def min_required_equity(
    *,
    risk_pct: float,
    min_notional: float,
    stop_dist_pct: float,
    cost_rt_pct: float,
    max_notional_pct: float,
) -> float:
    """Minimum equity required for a trade to satisfy min notional and caps."""

    rp = max(float(risk_pct), 0.0)
    min_notional_val = max(float(min_notional), 0.0)
    denom = float(stop_dist_pct) + float(cost_rt_pct)
    cap_mult = max(float(max_notional_pct), 0.0)
    if rp <= 0.0 or denom <= 0.0 or cap_mult <= 0.0:
        return float("inf")
    eq_risk = (min_notional_val * denom) / rp
    eq_cap = min_notional_val / cap_mult
    return float(max(eq_risk, eq_cap))


def explain_reject(trade_context: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic reject-feasibility explanation payload."""

    equity = float(trade_context.get("equity", 0.0))
    min_notional = float(trade_context.get("min_notional", 0.0))
    stop_dist_pct = float(trade_context.get("stop_dist_pct", 0.0))
    cost_rt_pct = float(trade_context.get("cost_rt_pct", 0.0))
    max_notional_pct = float(trade_context.get("max_notional_pct", 1.0))
    risk_pct_used = float(trade_context.get("risk_pct_used", 0.0))
    required_risk = min_required_risk_pct(
        equity=equity,
        min_notional=min_notional,
        stop_dist_pct=stop_dist_pct,
        cost_rt_pct=cost_rt_pct,
        max_notional_pct=max_notional_pct,
    )
    required_equity = min_required_equity(
        risk_pct=risk_pct_used if risk_pct_used > 0 else max(required_risk, 1e-12),
        min_notional=min_notional,
        stop_dist_pct=stop_dist_pct,
        cost_rt_pct=cost_rt_pct,
        max_notional_pct=max_notional_pct,
    )
    return {
        "equity": float(equity),
        "risk_pct_used": float(risk_pct_used),
        "min_notional": float(min_notional),
        "stop_dist_pct": float(stop_dist_pct),
        "cost_rt_pct": float(cost_rt_pct),
        "max_notional_pct": float(max_notional_pct),
        "minimum_required_risk_pct": float(required_risk) if np.isfinite(required_risk) else float("inf"),
        "minimum_required_equity": float(required_equity) if np.isfinite(required_equity) else float("inf"),
    }

