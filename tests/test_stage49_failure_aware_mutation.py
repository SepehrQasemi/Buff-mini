from __future__ import annotations

from buffmini.stage49.self_learning3 import failure_aware_mutation


def test_stage49_failure_aware_mutation_uses_motifs() -> None:
    assert (
        failure_aware_mutation({"raw_candidate_count": 0, "failure_motif_tags": ["NO_RAW_SIGNAL"]})
        == "widen_context_and_expand_grammar"
    )
    assert (
        failure_aware_mutation({"raw_candidate_count": 4, "failure_motif_tags": ["REJECT::BAD_RR"]})
        == "alter_geometry_and_invalidation"
    )
    assert (
        failure_aware_mutation({"raw_candidate_count": 4, "failure_motif_tags": ["REJECT::COST_DRAG"]})
        == "prioritize_high_edge_per_trade_setups"
    )

