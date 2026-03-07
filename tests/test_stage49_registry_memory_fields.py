from __future__ import annotations

from buffmini.stage49.self_learning3 import expand_registry_rows_v3


def test_stage49_registry_rows_include_required_memory_fields() -> None:
    rows = expand_registry_rows_v3(
        base_rows=[],
        seed=42,
        run_id="run42",
        stage47_counts={"structure_pullback_continuation": 3},
        stage48={"stage_a_survivors_after": 2, "stage_b_survivors_after": 1, "strict_direct_survivors_before": 4, "net_return_after_cost_mean": 0.001},
    )
    assert len(rows) >= 1
    row = rows[0]
    for key in (
        "run_id",
        "seed",
        "module_name",
        "family_name",
        "setup_name",
        "context_name",
        "feature_subset_signature",
        "threshold_config",
        "raw_candidate_count",
        "shortlisted_count",
        "activation_rate",
        "final_trade_count",
        "exp_lcb",
        "top_reject_reason",
        "failure_motif_tags",
        "elite",
        "mutation_origin",
        "runtime_cost",
        "contribution_summary",
    ):
        assert key in row

