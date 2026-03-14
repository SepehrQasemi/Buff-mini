from __future__ import annotations

import pandas as pd

from buffmini.research.learning import build_failure_taxonomy, derive_search_feedback
from buffmini.stage70 import generate_expanded_candidates


def test_stage82_feedback_prioritizes_promising_families() -> None:
    ranking = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "family": "failed_breakout_reversal",
                "candidate_class": "promising_but_unproven",
                "aggregate_risk": 0.35,
                "overlap_duplication_risk": 0.10,
                "cost_fragility_risk": 0.20,
                "regime_concentration_risk": 0.30,
                "clustering_risk": 0.20,
                "thin_evidence_risk": 0.20,
                "trade_density_risk": 0.20,
            },
            {
                "candidate_id": "b",
                "family": "structure_pullback_continuation",
                "candidate_class": "rejected",
                "aggregate_risk": 0.70,
                "overlap_duplication_risk": 0.60,
                "cost_fragility_risk": 0.60,
                "regime_concentration_risk": 0.70,
                "clustering_risk": 0.70,
                "thin_evidence_risk": 0.60,
                "trade_density_risk": 0.70,
            },
        ]
    )
    taxonomy = build_failure_taxonomy(
        ranking_cards=ranking,
        stage67_summary={"status": "PARTIAL", "blocker_reason": "insufficient_trades"},
        stage81_summary={"transfer_matrix_rows": 1, "transfer_class_counts": {"source_local": 1}},
    )
    feedback = derive_search_feedback(ranking_cards=ranking, failure_taxonomy=taxonomy)
    assert feedback["family_priority_adjustments"]["failed_breakout_reversal"] > feedback["family_priority_adjustments"]["structure_pullback_continuation"]
    assert "broaden_trade_density" in feedback["threshold_guidance"]

    baseline = generate_expanded_candidates(
        discovery_timeframes=["1h"],
        budget_mode_selected="search",
        active_families=["failed_breakout_reversal", "structure_pullback_continuation"],
        min_search_candidates=40,
    )
    adapted = generate_expanded_candidates(
        discovery_timeframes=["1h"],
        budget_mode_selected="search",
        active_families=["failed_breakout_reversal", "structure_pullback_continuation"],
        failure_feedback=feedback,
        min_search_candidates=40,
    )
    baseline_top = baseline.head(10)["family"].astype(str).value_counts().to_dict()
    adapted_top = adapted.head(10)["family"].astype(str).value_counts().to_dict()
    assert adapted_top.get("failed_breakout_reversal", 0) >= baseline_top.get("failed_breakout_reversal", 0)
