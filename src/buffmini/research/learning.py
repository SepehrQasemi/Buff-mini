"""Failure-driven search learning with reproducible feedback artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buffmini.utils.hashing import stable_hash


FAILURE_TAXONOMY = (
    "low_trade_count",
    "cost_fragile",
    "transfer_fail",
    "walkforward_fail",
    "perturbation_fail",
    "clustering_fail",
    "regime_overfit",
    "evidence_thin",
)


def build_failure_taxonomy(
    *,
    ranking_cards: pd.DataFrame,
    stage67_summary: dict[str, Any],
    stage81_summary: dict[str, Any],
) -> dict[str, int]:
    frame = ranking_cards.copy() if isinstance(ranking_cards, pd.DataFrame) else pd.DataFrame()
    counts = {key: 0 for key in FAILURE_TAXONOMY}
    if not frame.empty:
        counts["cost_fragile"] = int((pd.to_numeric(frame.get("cost_fragility_risk", 0.0), errors="coerce").fillna(0.0) >= 0.55).sum())
        counts["regime_overfit"] = int((pd.to_numeric(frame.get("regime_concentration_risk", 0.0), errors="coerce").fillna(0.0) >= 0.70).sum())
        counts["clustering_fail"] = int((pd.to_numeric(frame.get("clustering_risk", 0.0), errors="coerce").fillna(0.0) >= 0.60).sum())
        counts["evidence_thin"] = int((pd.to_numeric(frame.get("thin_evidence_risk", 0.0), errors="coerce").fillna(0.0) >= 0.55).sum())
        counts["low_trade_count"] = int((pd.to_numeric(frame.get("trade_density_risk", 0.0), errors="coerce").fillna(0.0) >= 0.70).sum())
    if str(stage67_summary.get("status", "")) != "SUCCESS":
        counts["walkforward_fail"] += 1
    if "insufficient" in str(stage67_summary.get("blocker_reason", "")).lower():
        counts["low_trade_count"] += 1
    if int(stage81_summary.get("transfer_matrix_rows", 0)) > 0:
        transfer_fail = 0
        for key, value in (stage81_summary.get("transfer_class_counts") or {}).items():
            if str(key) in {"source_local", "regime_local", "not_transferable"}:
                transfer_fail += int(value)
        counts["transfer_fail"] += int(transfer_fail)
    if "cross_perturbation_passed" in str(stage67_summary):
        pass
    return counts


def build_edge_inventory(ranking_cards: pd.DataFrame) -> list[dict[str, Any]]:
    frame = ranking_cards.copy() if isinstance(ranking_cards, pd.DataFrame) else pd.DataFrame()
    if frame.empty or "family" not in frame.columns:
        return []
    out: list[dict[str, Any]] = []
    grouped = frame.groupby("family", dropna=False)
    for family, group in grouped:
        class_counts = group.get("candidate_class", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
        out.append(
            {
                "family": str(family),
                "candidate_count": int(len(group)),
                "promising_count": int(class_counts.get("promising_but_unproven", 0)),
                "validated_count": int(class_counts.get("validated_candidate", 0)),
                "mean_rank_score": float(pd.to_numeric(group.get("rank_score", 0.0), errors="coerce").fillna(0.0).mean()),
                "mean_aggregate_risk": float(pd.to_numeric(group.get("aggregate_risk", 0.0), errors="coerce").fillna(0.0).mean()),
            }
        )
    return sorted(out, key=lambda row: (-float(row["promising_count"]), -float(row["mean_rank_score"]), str(row["family"])))


def derive_search_feedback(
    *,
    ranking_cards: pd.DataFrame,
    failure_taxonomy: dict[str, int],
) -> dict[str, Any]:
    frame = ranking_cards.copy() if isinstance(ranking_cards, pd.DataFrame) else pd.DataFrame()
    family_priority_adjustments: dict[str, float] = {}
    threshold_guidance: list[str] = []
    if not frame.empty and "family" in frame.columns:
        grouped = frame.groupby("family", dropna=False)
        for family, group in grouped:
            promising = float((group.get("candidate_class", pd.Series(dtype=str)).astype(str) == "promising_but_unproven").mean())
            risk = float(pd.to_numeric(group.get("aggregate_risk", 0.0), errors="coerce").fillna(0.0).mean())
            overlap = float(pd.to_numeric(group.get("overlap_duplication_risk", 0.0), errors="coerce").fillna(0.0).mean())
            family_priority_adjustments[str(family)] = float(round((promising * 0.18) - (risk * 0.10) - (overlap * 0.05), 6))
    if int(failure_taxonomy.get("low_trade_count", 0)) > 0:
        threshold_guidance.append("broaden_trade_density")
    if int(failure_taxonomy.get("cost_fragile", 0)) > 0:
        threshold_guidance.append("raise_cost_edge_floor")
    if int(failure_taxonomy.get("clustering_fail", 0)) > 0:
        threshold_guidance.append("diversify_mechanism_mix")
    if int(failure_taxonomy.get("transfer_fail", 0)) > 0:
        threshold_guidance.append("favor_lower_transfer_risk")
    payload = {
        "family_priority_adjustments": family_priority_adjustments,
        "threshold_guidance": threshold_guidance,
    }
    payload["feedback_hash"] = stable_hash(payload, length=16)
    return payload


def build_traceable_learning_loop(
    *,
    failure_taxonomy: dict[str, int],
    ranking_cards: pd.DataFrame,
    stage92_summary: dict[str, Any] | None = None,
    success_inventory: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Produce deterministic, traceable search refinement guidance."""

    feedback = derive_search_feedback(
        ranking_cards=ranking_cards,
        failure_taxonomy=failure_taxonomy,
    )
    transfer_signal = dict((stage92_summary or {}).get("transfer_class_counts", {}))
    success_inventory = list(success_inventory or [])
    success_weight = float(min(1.0, len(success_inventory) / 10.0))
    adaptation_steps: list[dict[str, Any]] = []
    if int(failure_taxonomy.get("low_trade_count", 0)) > 0:
        adaptation_steps.append(
            {
                "action": "broaden_trade_density_constraints",
                "reason": "low_trade_count",
                "magnitude": float(round(min(1.0, failure_taxonomy["low_trade_count"] / 50.0), 6)),
            }
        )
    if int(failure_taxonomy.get("clustering_fail", 0)) > 0:
        adaptation_steps.append(
            {
                "action": "increase_similarity_penalty",
                "reason": "clustering_fail",
                "magnitude": float(round(min(1.0, failure_taxonomy["clustering_fail"] / 100.0), 6)),
            }
        )
    if int(failure_taxonomy.get("transfer_fail", 0)) > 0 or int(transfer_signal.get("not_transferable", 0)) > 0:
        adaptation_steps.append(
            {
                "action": "downweight_transfer_fragile_families",
                "reason": "transfer_fail",
                "magnitude": float(
                    round(
                        min(
                            1.0,
                            (
                                int(failure_taxonomy.get("transfer_fail", 0))
                                + int(transfer_signal.get("not_transferable", 0))
                            )
                            / 25.0,
                        ),
                        6,
                    )
                ),
            }
        )
    if int(failure_taxonomy.get("walkforward_fail", 0)) > 0:
        adaptation_steps.append(
            {
                "action": "favor_forward_stable_regimes",
                "reason": "walkforward_fail",
                "magnitude": float(round(min(1.0, failure_taxonomy["walkforward_fail"] / 10.0), 6)),
            }
        )
    if success_weight > 0.0:
        adaptation_steps.append(
            {
                "action": "success_weighted_mechanism_retention",
                "reason": "promising_inventory_present",
                "magnitude": float(round(success_weight, 6)),
            }
        )
    payload = {
        "failure_taxonomy": dict(failure_taxonomy),
        "search_feedback": feedback,
        "transfer_signal": transfer_signal,
        "success_inventory_count": int(len(success_inventory)),
        "adaptation_steps": adaptation_steps,
    }
    payload["learning_trace_hash"] = stable_hash(payload, length=16)
    return payload
