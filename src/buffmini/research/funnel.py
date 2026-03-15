"""Candidate funnel pressure diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from buffmini.research.campaign import evaluate_scope_campaign, select_campaign_families
from buffmini.stage51.scope import resolve_research_scope
from buffmini.utils.hashing import stable_hash


def analyze_funnel_pressure(
    config: dict[str, Any],
    *,
    feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scope = resolve_research_scope(config)
    families = select_campaign_families(feedback, limit=6)
    symbol = str((scope.get("primary_symbols") or ["BTC/USDT"])[0])
    timeframe = str((scope.get("discovery_timeframes") or ["1h"])[2 if len(scope.get("discovery_timeframes", [])) >= 3 else 0])
    summary = evaluate_scope_campaign(
        config=config,
        symbol=symbol,
        timeframe=timeframe,
        families=families,
        candidate_limit=12,
        requested_mode="exploration",
        auto_pin_resolved_end=False,
        relax_continuity=True,
        evaluate_transfer=False,
    )
    evaluations = list(summary.get("evaluations", []))
    hierarchy_counts = Counter(str(row.get("candidate_hierarchy", "junk")) for row in evaluations)
    final_class_counts = Counter(str(row.get("final_class", "rejected")) for row in evaluations)
    gate_heatmap = _build_gate_heatmap(evaluations)
    near_miss_inventory = sorted(
        [
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "family": str(row.get("family", "")),
                "candidate_hierarchy": str(row.get("candidate_hierarchy", "")),
                "first_death_stage": str(row.get("first_death_stage", "")),
                "death_reason": str(row.get("death_reason", "")),
                "near_miss_distance": float(row.get("near_miss_distance", 0.0)),
            }
            for row in evaluations
            if 0.0 < float(row.get("near_miss_distance", 0.0)) <= 1.25
        ],
        key=lambda row: (float(row["near_miss_distance"]), row["candidate_id"]),
    )
    diagnosis = diagnose_funnel_pressure(
        evaluations=evaluations,
        blocked_count=int(summary.get("blocked_count", 0)),
        candidate_count=int(summary.get("candidate_count", 0)),
    )
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candidate_count": int(summary.get("candidate_count", 0)),
        "promising_count": int(summary.get("promising_count", 0)),
        "validated_count": int(summary.get("validated_count", 0)),
        "robust_count": int(summary.get("robust_count", 0)),
        "blocked_count": int(summary.get("blocked_count", 0)),
        "candidate_hierarchy_counts": dict(hierarchy_counts),
        "candidate_class_counts": dict(final_class_counts),
        "near_miss_inventory": near_miss_inventory[:20],
        "gate_heatmap": gate_heatmap,
        "diagnosis": diagnosis,
        "evaluations": evaluations,
        "summary_hash": stable_hash(
            {
                "candidate_count": int(summary.get("candidate_count", 0)),
                "promising_count": int(summary.get("promising_count", 0)),
                "diagnosis": diagnosis,
                "gate_heatmap": gate_heatmap,
            },
            length=16,
        ),
    }


def diagnose_funnel_pressure(
    *,
    evaluations: list[dict[str, Any]],
    blocked_count: int,
    candidate_count: int,
) -> dict[str, Any]:
    if int(blocked_count) > 0:
        culprit = "data_limited"
    else:
        near_miss = [
            float(row.get("near_miss_distance", 9.0))
            for row in evaluations
            if str(row.get("candidate_hierarchy", "")) in {"promising_but_unproven", "interesting_but_fragile"}
        ]
        avg_near_miss = sum(near_miss) / len(near_miss) if near_miss else 9.0
        if avg_near_miss <= 0.55:
            culprit = "funnel_pressure_over_tight"
        elif avg_near_miss <= 1.25:
            culprit = "generator_depth_limited"
        else:
            culprit = "ranking_and_generator_both_weak"
    return {
        "dominant_culprit": culprit,
        "candidate_count": int(candidate_count),
        "blocked_count": int(blocked_count),
    }


def _build_gate_heatmap(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], int] = defaultdict(int)
    for row in evaluations:
        family = str(row.get("family", ""))
        stage = str(row.get("first_death_stage", "survived"))
        buckets[(family, stage)] += 1
    return [
        {
            "family": family,
            "gate": gate,
            "count": int(count),
        }
        for (family, gate), count in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
