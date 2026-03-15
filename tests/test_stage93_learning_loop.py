from __future__ import annotations

import pandas as pd

from buffmini.research.learning import build_traceable_learning_loop


def test_stage93_learning_loop_is_traceable_and_deterministic() -> None:
    ranking_cards = pd.DataFrame(
        [
            {
                "family": "structure_pullback_continuation",
                "candidate_class": "promising_but_unproven",
                "rank_score": 0.6,
                "aggregate_risk": 0.4,
                "overlap_duplication_risk": 0.8,
                "cost_fragility_risk": 0.6,
                "trade_density_risk": 0.7,
                "regime_concentration_risk": 0.5,
                "clustering_risk": 0.7,
                "thin_evidence_risk": 0.5,
            }
        ]
    )
    failure_taxonomy = {
        "low_trade_count": 3,
        "cost_fragile": 2,
        "transfer_fail": 1,
        "walkforward_fail": 1,
        "perturbation_fail": 0,
        "clustering_fail": 4,
        "regime_overfit": 0,
        "evidence_thin": 1,
    }
    stage92 = {"transfer_class_counts": {"not_transferable": 2}}
    loop_a = build_traceable_learning_loop(
        failure_taxonomy=failure_taxonomy,
        ranking_cards=ranking_cards,
        stage92_summary=stage92,
        success_inventory=[{"candidate_id": "c1"}],
    )
    loop_b = build_traceable_learning_loop(
        failure_taxonomy=failure_taxonomy,
        ranking_cards=ranking_cards,
        stage92_summary=stage92,
        success_inventory=[{"candidate_id": "c1"}],
    )
    assert loop_a["learning_trace_hash"] == loop_b["learning_trace_hash"]
    assert loop_a["adaptation_steps"]
