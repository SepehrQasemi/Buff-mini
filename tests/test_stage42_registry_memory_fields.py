from __future__ import annotations

from buffmini.stage42.self_learning2 import expand_registry_rows, family_memory_summary


def test_stage42_registry_memory_fields_present() -> None:
    base = [
        {
            "run_id": "r1",
            "family": "funding",
            "feature_subset_signature": "family::funding",
            "threshold_configuration": {"activation_threshold": 0.0},
            "raw_signal_count": 5,
            "activation_rate": 0.1,
            "final_trade_count": 1,
            "exp_lcb": 0.01,
            "top_reject_reason": "none",
            "cost_gate_fail_rate": 0.2,
            "feasibility_fail_rate": 0.1,
            "elite": True,
            "failure_motif_tags": ["REJECT::NONE"],
        }
    ]
    rows = expand_registry_rows(base, seed=42, raw_candidate_count=10, shortlisted_count=4)
    assert len(rows) == 1
    row = rows[0]
    for key in (
        "seed",
        "context",
        "raw_candidate_count",
        "shortlisted_count",
        "mutation_origin",
        "mutation_guidance",
        "failure_motif_tags",
    ):
        assert key in row

    memory = family_memory_summary(rows)
    assert "family_weights" in memory
    assert "top_configs_per_family" in memory
    assert "recurring_failure_motifs" in memory

