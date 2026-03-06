from __future__ import annotations

from pathlib import Path

from buffmini.stage37.self_learning import (
    apply_elite_flags,
    load_learning_registry,
    save_learning_registry,
    select_elites_deterministic,
    upsert_learning_registry_entry,
)
from buffmini.stage38.audit import build_failure_aware_registry_rows


def test_stage38_self_learning_registry_populates_on_zero_trade_case(tmp_path: Path) -> None:
    activation_payload = {
        "chosen_threshold": 0.0,
        "hunt": {
            "overall": {
                "raw_signal_count": 24,
                "post_threshold_count": 24,
                "post_cost_gate_count": 24,
                "post_feasibility_count": 24,
                "composer_signal_count": 0,
                "final_trade_count": 0.0,
                "activation_rate": 0.0,
                "avg_context_quality": 0.0,
                "top_reject_reasons": {},
            },
            "per_family": {},
        },
    }
    rows = build_failure_aware_registry_rows(activation_payload=activation_payload, run_id="stage38_r0")
    assert len(rows) == 1
    row = rows[0]
    assert row["family"] == "__all__"
    assert row["status"] == "dead_end"
    assert row["final_trade_count"] == 0
    assert row["failure_motif_tags"]

    registry_path = tmp_path / "learning_registry.json"
    upsert_learning_registry_entry(registry_path, row)
    stored = load_learning_registry(registry_path)
    assert len(stored) == 1
    assert stored[0]["failure_motif_tags"]


def test_stage38_self_learning_elite_flags_are_persistable(tmp_path: Path) -> None:
    registry_path = tmp_path / "learning_registry.json"
    base_rows = [
        {
            "run_id": "r1",
            "generation": 0,
            "family": "funding",
            "feature_subset_signature": "family::funding",
            "threshold_configuration": {"activation_threshold": 0.0},
            "raw_signal_count": 10,
            "activation_rate": 0.1,
            "top_reject_reason": "none",
            "cost_gate_fail_rate": 0.1,
            "feasibility_fail_rate": 0.1,
            "final_trade_count": 3,
            "exp_lcb": 0.01,
            "stability_score": 0.1,
            "status": "active",
            "elite": False,
            "failure_motif_tags": ["REJECT::NONE"],
        },
        {
            "run_id": "r2",
            "generation": 0,
            "family": "flow",
            "feature_subset_signature": "family::flow",
            "threshold_configuration": {"activation_threshold": 0.0},
            "raw_signal_count": 8,
            "activation_rate": 0.02,
            "top_reject_reason": "SIZE_TOO_SMALL",
            "cost_gate_fail_rate": 0.3,
            "feasibility_fail_rate": 0.2,
            "final_trade_count": 0,
            "exp_lcb": -0.01,
            "stability_score": 0.02,
            "status": "dead_end",
            "elite": False,
            "failure_motif_tags": ["REJECT::SIZE_TOO_SMALL"],
        },
    ]
    save_learning_registry(registry_path, base_rows)
    rows = load_learning_registry(registry_path)
    elites = select_elites_deterministic(rows, top_k=1)
    flagged = apply_elite_flags(rows, elites)
    save_learning_registry(registry_path, flagged)
    stored = load_learning_registry(registry_path)
    assert sum(1 for row in stored if bool(row.get("elite", False))) == 1

