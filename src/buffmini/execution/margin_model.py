"""Canonical margin + exposure cap model for execution feasibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyCaps:
    """Policy caps interpreted as equity-relative limits where possible."""

    max_notional_pct_of_equity: float = 1.0
    max_gross_exposure_mult: float = 1.0
    absolute_max_notional: float = 0.0
    margin_alloc_limit: float = 1.0


def compute_margin_required(
    notional: float,
    leverage: float,
    fees_estimate: float,
    buffer: float,
) -> float:
    """Required margin for one order using deterministic, scale-consistent math."""

    notion = abs(float(notional))
    lev = max(float(leverage), 1e-12)
    fees = max(float(fees_estimate), 0.0)
    margin_buffer = max(float(buffer), 0.0)
    return float((notion / lev) + (notion * fees) + (notion * margin_buffer))


def apply_exposure_caps(
    desired_notional: float,
    policy_caps: PolicyCaps,
    current_exposure: float,
    equity: float,
) -> tuple[float, str, dict[str, Any]]:
    """Apply policy caps before margin checks."""

    desired = max(0.0, float(desired_notional))
    eq = max(0.0, float(equity))
    used = max(0.0, float(current_exposure))
    max_pct = max(0.0, float(policy_caps.max_notional_pct_of_equity))
    max_gross_mult = max(0.0, float(policy_caps.max_gross_exposure_mult))
    abs_max = max(0.0, float(policy_caps.absolute_max_notional))

    candidates = []
    if max_pct > 0:
        candidates.append(eq * max_pct)
    if max_gross_mult > 0:
        candidates.append(eq * max_gross_mult)
    if abs_max > 0:
        candidates.append(abs_max)
    max_allowed = min(candidates) if candidates else 0.0
    available = max(0.0, max_allowed - used)
    capped = min(desired, available)
    reason = "POLICY_CAP_HIT" if capped + 1e-12 < desired else ""
    return float(capped), reason, {
        "desired_notional": float(desired),
        "max_allowed_notional": float(max_allowed),
        "available_notional": float(available),
        "current_exposure": float(used),
    }


def is_trade_feasible(
    equity: float,
    capped_notional: float,
    leverage: float,
    margin_required: float,
    policy_caps: PolicyCaps,
) -> tuple[bool, str, dict[str, Any]]:
    """Final feasibility after caps + margin requirements."""

    eq = max(0.0, float(equity))
    notion = max(0.0, float(capped_notional))
    margin_req = max(0.0, float(margin_required))
    lev = max(float(leverage), 1e-12)
    margin_alloc_limit = max(0.0, float(policy_caps.margin_alloc_limit))
    margin_limit = float(eq * margin_alloc_limit)

    if notion <= 0.0:
        return False, "POLICY_CAP_HIT", {
            "equity": float(eq),
            "notional": float(notion),
            "leverage": float(lev),
            "margin_required": float(margin_req),
            "margin_limit": float(margin_limit),
            "margin_alloc_limit": float(margin_alloc_limit),
        }
    if margin_req > margin_limit + 1e-12:
        return False, "MARGIN_FAIL", {
            "equity": float(eq),
            "notional": float(notion),
            "leverage": float(lev),
            "margin_required": float(margin_req),
            "margin_limit": float(margin_limit),
            "margin_alloc_limit": float(margin_alloc_limit),
        }
    return True, "", {
        "equity": float(eq),
        "notional": float(notion),
        "leverage": float(lev),
        "margin_required": float(margin_req),
        "margin_limit": float(margin_limit),
        "margin_alloc_limit": float(margin_alloc_limit),
    }
