from __future__ import annotations

from pathlib import Path

from buffmini.validation import (
    build_metric_evidence,
    decision_evidence_guard,
    validate_metric_evidence,
    validate_metric_evidence_batch,
)


def test_metric_evidence_schema_validates() -> None:
    record = build_metric_evidence(
        candidate_id="c1",
        run_id="r1",
        config_hash="cfg_hash",
        data_hash="data_hash",
        seed=42,
        metric_name="exp_lcb",
        metric_value=0.01,
        metric_source_type="real_replay",
        artifact_path="docs/stage53_summary.json",
        stage_origin="stage53",
        used_for_decision=True,
    )
    out = validate_metric_evidence(record, repo_root=Path("."))
    assert out["valid"] is True


def test_decision_guard_blocks_proxy_and_missing_real_sources() -> None:
    records = [
        build_metric_evidence(
            candidate_id="c1",
            run_id="r1",
            config_hash="cfg_hash",
            data_hash="data_hash",
            seed=42,
            metric_name="exp_lcb",
            metric_value=0.01,
            metric_source_type="proxy_only",
            artifact_path="docs/stage53_summary.json",
            stage_origin="stage53",
            used_for_decision=True,
        )
    ]
    schema = validate_metric_evidence_batch(records, repo_root=Path("."))
    assert schema["valid"] is True
    guard = decision_evidence_guard(
        records,
        required_real_sources=["real_replay", "real_walkforward"],
        repo_root=Path("."),
    )
    assert guard["allowed"] is False
    assert "real_replay" in guard["missing_real_sources"] or "real_walkforward" in guard["missing_real_sources"]
    assert "exp_lcb" in guard["blocked_decision_metrics"]
