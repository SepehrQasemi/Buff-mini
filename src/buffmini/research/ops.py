"""Research operations helpers for run registry and review checklists."""

from __future__ import annotations

from typing import Any


def build_run_registry_entry(
    *,
    run_id: str,
    mode_summary: dict[str, Any],
    stage80_summary: dict[str, Any],
    stage81_summary: dict[str, Any],
    stage82_summary: dict[str, Any],
) -> dict[str, Any]:
    level = int(stage80_summary.get("level_reached", 0))
    transfer_classes = dict(stage81_summary.get("transfer_class_counts", {}))
    outcome_class = "rejected"
    if level >= 3:
        outcome_class = "validated_candidate"
    elif level >= 1 or int(transfer_classes.get("partially_transferable", 0)) > 0:
        outcome_class = "promising_but_unproven"
    return {
        "run_id": str(run_id),
        "run_purpose": str(mode_summary.get("run_type", mode_summary.get("mode", "exploration"))),
        "mode": str(mode_summary.get("mode", "exploration")),
        "dataset_scope": {
            "resolved_end_ts": str(mode_summary.get("resolved_end_ts", "")),
            "data_scope_hash": str(mode_summary.get("data_scope_hash", "")),
        },
        "seed_set": list(mode_summary.get("seed_bundle", [])),
        "gating_level": int(level),
        "outcome_class": outcome_class,
        "canonical_status": str(mode_summary.get("canonical_status", "")),
        "interpretation_allowed": bool(mode_summary.get("interpretation_allowed", False)),
        "feedback_hash": str((stage82_summary.get("search_feedback") or {}).get("feedback_hash", "")),
    }


def build_human_review_checklist(
    *,
    mode_summary: dict[str, Any],
    stage67_summary: dict[str, Any],
    stage80_summary: dict[str, Any],
    stage81_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "item": "evaluation_mode_ready",
            "passed": bool(mode_summary.get("interpretation_allowed", False)),
            "reason": "" if bool(mode_summary.get("interpretation_allowed", False)) else ",".join(mode_summary.get("blocked_reasons", [])),
        },
        {
            "item": "runtime_truth_clear",
            "passed": not bool(stage67_summary.get("runtime_truth_blocked", False)),
            "reason": str(stage67_summary.get("runtime_truth_reason", "")),
        },
        {
            "item": "robustness_level_documented",
            "passed": int(stage80_summary.get("level_reached", 0)) >= 1,
            "reason": str(stage80_summary.get("stop_reason", "")),
        },
        {
            "item": "transfer_matrix_present",
            "passed": int(stage81_summary.get("transfer_matrix_rows", 0)) > 0,
            "reason": "missing_transfer_matrix" if int(stage81_summary.get("transfer_matrix_rows", 0)) <= 0 else "",
        },
    ]
