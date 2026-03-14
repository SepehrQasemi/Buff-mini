from __future__ import annotations

from buffmini.research.ops import build_human_review_checklist, build_run_registry_entry


def test_stage83_ops_builds_registry_and_checklist() -> None:
    registry = build_run_registry_entry(
        run_id="r1",
        mode_summary={"mode": "evaluation", "run_type": "evaluation", "resolved_end_ts": "2025-01-01T00:00:00Z", "data_scope_hash": "abc", "seed_bundle": [11, 19], "canonical_status": "CANONICAL", "interpretation_allowed": True},
        stage80_summary={"level_reached": 2},
        stage81_summary={"transfer_class_counts": {"partially_transferable": 1}},
        stage82_summary={"search_feedback": {"feedback_hash": "xyz"}},
    )
    checklist = build_human_review_checklist(
        mode_summary={"interpretation_allowed": True, "blocked_reasons": []},
        stage67_summary={"runtime_truth_blocked": False, "runtime_truth_reason": ""},
        stage80_summary={"level_reached": 2, "stop_reason": "full_robustness_not_met"},
        stage81_summary={"transfer_matrix_rows": 1},
    )
    assert registry["outcome_class"] == "promising_but_unproven"
    assert registry["feedback_hash"] == "xyz"
    assert all("item" in row and "passed" in row for row in checklist)
