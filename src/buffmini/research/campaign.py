"""Helpers for bounded serious edge campaigns."""

from __future__ import annotations

from typing import Any


DEFAULT_CAMPAIGN_FAMILIES = (
    "structure_pullback_continuation",
    "failed_breakout_reversal",
    "volatility_regime_transition",
    "exhaustion_mean_reversion",
)


def select_campaign_families(feedback: dict[str, Any] | None = None, *, limit: int = 4) -> list[str]:
    adjustments = dict((feedback or {}).get("family_priority_adjustments", {}))
    ordered = list(DEFAULT_CAMPAIGN_FAMILIES)
    if adjustments:
        ordered = sorted(
            DEFAULT_CAMPAIGN_FAMILIES,
            key=lambda family: (-float(adjustments.get(family, 0.0)), DEFAULT_CAMPAIGN_FAMILIES.index(family)),
        )
    return [str(family) for family in ordered[: max(1, int(limit))]]


def classify_campaign_outcome(
    *,
    edge_inventory: list[dict[str, Any]],
    evaluated_assets: int,
    blocked_assets: int,
) -> str:
    robust = sum(1 for row in edge_inventory if str(row.get("final_class", "")) == "robust_candidate")
    promising = sum(1 for row in edge_inventory if str(row.get("final_class", "")) == "promising_but_unproven")
    if evaluated_assets <= 0 and blocked_assets > 0:
        return "system_blocked_uninterpretable"
    if robust > 0:
        return "robust_edge_candidates_present"
    if promising > 0:
        return "weak_promising_signs_need_refinement"
    return "honest_no_robust_edge_survived"
