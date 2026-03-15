"""Stage-99 candidate quality acceleration and truth review."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from buffmini.research.campaign import evaluate_scope_campaign
from buffmini.stage51.scope import resolve_research_scope
from buffmini.utils.hashing import stable_hash


def evaluate_candidate_quality_acceleration(
    config: dict[str, Any],
    *,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    candidate_limit: int = 6,
) -> dict[str, Any]:
    scope = resolve_research_scope(config)
    families = list(scope.get("active_setup_families") or [])
    before = evaluate_scope_campaign(
        config=config,
        symbol=symbol,
        timeframe=timeframe,
        families=families,
        candidate_limit=int(candidate_limit),
        requested_mode="exploration",
        auto_pin_resolved_end=False,
        relax_continuity=True,
        evaluate_transfer=True,
        ranking_profile="stage95_usefulness_push",
    )
    after = evaluate_scope_campaign(
        config=config,
        symbol=symbol,
        timeframe=timeframe,
        families=families,
        candidate_limit=int(candidate_limit),
        requested_mode="exploration",
        auto_pin_resolved_end=False,
        relax_continuity=True,
        evaluate_transfer=True,
        ranking_profile="stage99_quality_acceleration",
    )
    transition_rows = build_quality_transition_rows(before=before, after=after)
    gate_heatmap = _build_gate_heatmap(list(after.get("evaluations", [])))
    near_miss_inventory = [
        {
            "candidate_id": str(row.get("candidate_id", "")),
            "family": str(row.get("family", "")),
            "near_miss_distance": float(row.get("near_miss_distance", 0.0)),
            "death_reason": str(row.get("death_reason", "")),
        }
        for row in list(after.get("evaluations", []))
        if 0.0 < float(row.get("near_miss_distance", 0.0)) <= 1.0
    ]
    top_k_review = build_top_k_truth_review(after=after, top_k=min(8, int(candidate_limit)))
    candidate_hierarchy_counts = Counter(str(row.get("candidate_hierarchy", "")) for row in list(after.get("evaluations", [])))
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candidate_limit": int(candidate_limit),
        "before_profile": "stage95_usefulness_push",
        "after_profile": "stage99_quality_acceleration",
        "before_counts": _summary_counts(before),
        "after_counts": _summary_counts(after),
        "transition_rows": transition_rows,
        "gate_heatmap": gate_heatmap,
        "near_miss_inventory": near_miss_inventory,
        "candidate_hierarchy_counts": dict(candidate_hierarchy_counts),
        "top_k_truth_review": top_k_review,
        "stage99b_required": bool(int(after.get("promising_count", 0)) > 0 and int(after.get("validated_count", 0)) == 0),
        "stage99b_applied": True,
        "summary_hash": stable_hash(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "after_counts": _summary_counts(after),
                "transition_rows": transition_rows,
            },
            length=16,
        ),
    }


def build_quality_transition_rows(*, before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    before_lookup = {str(row.get("candidate_id", "")): row for row in list(before.get("evaluations", []))}
    after_lookup = {str(row.get("candidate_id", "")): row for row in list(after.get("evaluations", []))}
    ranked_before = {str(row.get("candidate_id", "")): row for row in getattr(before.get("ranked_frame"), "to_dict", lambda **_: [])(orient="records")}
    ranked_after = {str(row.get("candidate_id", "")): row for row in getattr(after.get("ranked_frame"), "to_dict", lambda **_: [])(orient="records")}
    rows: list[dict[str, Any]] = []
    for candidate_id in sorted(set(before_lookup) | set(after_lookup)):
        before_row = before_lookup.get(candidate_id, {})
        after_row = after_lookup.get(candidate_id, {})
        before_rank = float((ranked_before.get(candidate_id, {}) or {}).get("rank_score", 0.0))
        after_rank = float((ranked_after.get(candidate_id, {}) or {}).get("rank_score", 0.0))
        after_ranked = ranked_after.get(candidate_id, {}) or {}
        rows.append(
            {
                "candidate_id": candidate_id,
                "family": str((after_row or before_row).get("family", "")),
                "before_final_class": str(before_row.get("final_class", "absent")),
                "after_final_class": str(after_row.get("final_class", "absent")),
                "before_rank_score": before_rank,
                "after_rank_score": after_rank,
                "rank_delta": round(float(after_rank - before_rank), 6),
                "change_reason": explain_quality_reasons(dict(after_ranked)),
            }
        )
    return rows


def build_top_k_truth_review(*, after: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    evaluations = sorted(
        list(after.get("evaluations", [])),
        key=lambda row: (-float(row.get("rank_score", 0.0)), str(row.get("candidate_id", ""))),
    )[: max(1, int(top_k))]
    rows: list[dict[str, Any]] = []
    for row in evaluations:
        rescue_hint = "none"
        if float(row.get("cost_fragility_risk", 0.0)) >= 0.55:
            rescue_hint = "lower_cost_exposure"
        elif float(row.get("hold_sanity_risk", 0.0)) >= 0.55:
            rescue_hint = "shorter_hold_horizon"
        elif float(row.get("transfer_risk_prior", 0.0)) >= 0.45:
            rescue_hint = "tighten_participation_and_transfer_exposure"
        rows.append(
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "family": str(row.get("family", "")),
                "why_it_surfaced": explain_quality_reasons(row),
                "target_regime": str(row.get("expected_regime", "")),
                "replay_issue": str(row.get("death_reason", "")) if str(row.get("first_death_stage", "")) == "replay" else "",
                "transfer_issue": ",".join(list(row.get("transfer_diagnostics", []))) if str(row.get("transfer_classification", "")) not in {"transferable", "partially_transferable"} else "",
                "mc_issue": "" if bool(row.get("monte_carlo_passed", False)) else str(row.get("robustness_stop_reason", "")),
                "rescue_hint": rescue_hint,
            }
        )
    return rows


def explain_quality_reasons(candidate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if float(candidate.get("trade_quality_bonus", 0.0)) >= 0.08:
        reasons.append("strong_trade_quality_bonus")
    if float(candidate.get("usefulness_prior", 0.0)) >= 0.05:
        reasons.append("useful_trade_density_or_transfer_prior")
    if float(candidate.get("trade_density_risk", 1.0)) <= 0.45:
        reasons.append("acceptable_trade_density")
    if float(candidate.get("clustering_risk", 1.0)) <= 0.45:
        reasons.append("lower_clustering_risk")
    if float(candidate.get("thin_evidence_risk", 1.0)) <= 0.45:
        reasons.append("thicker_evidence_profile")
    if not reasons:
        reasons.append("no_quality_edge")
    return reasons


def _summary_counts(summary: dict[str, Any]) -> dict[str, int]:
    evaluations = list(summary.get("evaluations", []))
    return {
        "candidate_count": int(summary.get("candidate_count", 0)),
        "promising_count": int(summary.get("promising_count", 0)),
        "validated_count": int(summary.get("validated_count", 0)),
        "robust_count": int(summary.get("robust_count", 0)),
        "interesting_count": int(sum(1 for row in evaluations if str(row.get("candidate_hierarchy", "")) == "interesting_but_fragile")),
        "rejected_count": int(sum(1 for row in evaluations if str(row.get("final_class", "")) == "rejected")),
    }


def _build_gate_heatmap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        buckets[(str(row.get("family", "")), str(row.get("first_death_stage", "")))] += 1
    return [
        {"family": family, "gate": gate, "count": int(count)}
        for (family, gate), count in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
