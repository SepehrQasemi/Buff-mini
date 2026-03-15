from __future__ import annotations

from buffmini.research.verdict import derive_final_edge_verdict


def test_derive_final_edge_verdict_prefers_generator_insufficient_when_fragile_and_controls_fail() -> None:
    payloads = {
        "95": {"dead_weight_families": [{"family": "funding"}]},
        "96": {"rows": [{"canonical_usable": True}]},
        "100": {"truth_counts": {"replay_fragile_signal_only": 6, "data_blocks_interpretation": 0}},
        "101": {"candidate_beats_all_controls_count": 0, "candidate_beats_majority_controls_count": 0},
        "102": {"classification_counts": {"still_generator_weak": 2}},
    }
    summary = derive_final_edge_verdict(payloads)
    assert summary["final_edge_verdict"] == "GENERATOR_OR_SEARCH_FORMALISM_STILL_INSUFFICIENT"


def test_derive_final_edge_verdict_prefers_robust_when_rescueable_exists() -> None:
    payloads = {
        "95": {"dead_weight_families": []},
        "96": {"rows": [{"canonical_usable": True}]},
        "100": {"truth_counts": {"replay_fragile_signal_only": 0}},
        "101": {"candidate_beats_all_controls_count": 1, "candidate_beats_majority_controls_count": 1},
        "102": {"classification_counts": {"rescueable": 1}},
    }
    summary = derive_final_edge_verdict(payloads)
    assert summary["final_edge_verdict"] == "ROBUST_CANDIDATE_FOUND"
