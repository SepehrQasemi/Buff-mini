from __future__ import annotations

from pathlib import Path

from buffmini.stage37.self_learning import (
    LearningRegistryEntry,
    compute_family_exploration_weights,
    load_learning_registry,
    prune_features_by_contribution,
    select_elites_deterministic,
    upsert_learning_registry_entry,
)


def _entry(*, run_id: str, family: str, activation_rate: float, exp_lcb: float, trades: int, reason: str) -> LearningRegistryEntry:
    return LearningRegistryEntry(
        run_id=run_id,
        generation=0,
        family=family,
        feature_subset_signature=f"sig::{family}",
        threshold_configuration={"activation_threshold": 0.05},
        activation_rate=activation_rate,
        top_reject_reason=reason,
        cost_gate_fail_rate=0.2,
        feasibility_fail_rate=0.1,
        final_trade_count=trades,
        exp_lcb=exp_lcb,
        stability_score=activation_rate,
        status="active" if trades > 0 else "dead_end",
    )


def test_stage37_registry_schema_fields_present(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    upsert_learning_registry_entry(path, _entry(run_id="r1", family="funding", activation_rate=0.2, exp_lcb=0.01, trades=5, reason="SIZE_TOO_SMALL"))
    rows = load_learning_registry(path)
    assert len(rows) == 1
    row = rows[0]
    for key in (
        "family",
        "feature_subset_signature",
        "threshold_configuration",
        "activation_rate",
        "top_reject_reason",
        "cost_gate_fail_rate",
        "feasibility_fail_rate",
        "final_trade_count",
        "exp_lcb",
        "stability_score",
    ):
        assert key in row


def test_stage37_elites_selection_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    upsert_learning_registry_entry(path, _entry(run_id="r1", family="funding", activation_rate=0.2, exp_lcb=0.05, trades=10, reason="none"))
    upsert_learning_registry_entry(path, _entry(run_id="r2", family="flow", activation_rate=0.3, exp_lcb=0.01, trades=8, reason="SIZE_TOO_SMALL"))
    upsert_learning_registry_entry(path, _entry(run_id="r3", family="ls", activation_rate=0.1, exp_lcb=0.02, trades=4, reason="NO_EDGE"))
    rows = load_learning_registry(path)
    first = select_elites_deterministic(rows, top_k=2)
    second = select_elites_deterministic(rows, top_k=2)
    assert first == second


def test_stage37_dead_families_down_weighted() -> None:
    rows = [
        _entry(run_id="r1", family="dead_family", activation_rate=0.0, exp_lcb=-0.1, trades=0, reason="SIZE_TOO_SMALL").as_dict(),
        _entry(run_id="r2", family="dead_family", activation_rate=0.0, exp_lcb=-0.2, trades=0, reason="SIZE_TOO_SMALL").as_dict(),
        _entry(run_id="r3", family="active_family", activation_rate=0.3, exp_lcb=0.01, trades=6, reason="none").as_dict(),
    ]
    weights = compute_family_exploration_weights(rows)
    assert float(weights["dead_family"]) < float(weights["active_family"])


def test_stage37_feature_pruning_schema_stable() -> None:
    rows = [
        {"feature": "f1", "gain": 0.05},
        {"feature": "f1", "gain": 0.02},
        {"feature": "f2", "gain": -0.01},
        {"feature": "f3", "gain": 0.01},
    ]
    out = prune_features_by_contribution(rows, min_mean_gain=0.0, keep_top=2)
    assert "kept_features" in out
    assert "dropped_features" in out
    assert "contribution_summary" in out
    assert len(out["kept_features"]) <= 2
