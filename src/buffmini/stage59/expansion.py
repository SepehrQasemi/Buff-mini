"""Stage-59 conditional expansion planning."""

from __future__ import annotations

from typing import Any


def build_conditional_expansion(
    *,
    stage58_verdict: str,
    transfer_acceptable: bool = False,
    active_families: list[str],
    next_families: list[str] | None = None,
    oi_short_horizon_only: bool = True,
    paid_provider_optional: bool = True,
) -> dict[str, Any]:
    verdict = str(stage58_verdict)
    eligible = bool(transfer_acceptable) and verdict in {"WEAK_EDGE", "MEDIUM_EDGE"}
    if not eligible:
        status = "BLOCKED_WAITING_VALID_INPUTS" if verdict == "STALE_INPUTS" else "BLOCKED_RESEARCH_REVIEW_REQUIRED"
        return {
            "status": status,
            "legacy_status": "PARTIAL",
            "expansion_allowed": False,
            "next_families": [],
            "oi_mode": "short_horizon_only",
            "paid_provider_optional": bool(paid_provider_optional),
        }
    expansion_queue = [family for family in (next_families or []) if str(family) not in set(active_families)]
    return {
        "status": "EXPANSION_ALLOWED",
        "legacy_status": "SUCCESS",
        "expansion_allowed": True,
        "next_families": expansion_queue,
        "oi_mode": "short_horizon_only" if bool(oi_short_horizon_only) else "full_history_if_covered",
        "paid_provider_optional": bool(paid_provider_optional),
    }
