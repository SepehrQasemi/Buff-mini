from __future__ import annotations

from pathlib import Path

from buffmini.stage58 import assess_transfer_validation


def test_stage58_requires_stage57_pass() -> None:
    result = assess_transfer_validation(stage57_verdict="PARTIAL", primary_metrics={"exp_lcb": 0.01}, transfer_metrics=None)
    assert result["verdict"] == "PARTIAL"


def test_stage58_propagates_no_edge_from_stage57() -> None:
    result = assess_transfer_validation(stage57_verdict="NO_EDGE_IN_SCOPE", primary_metrics={"exp_lcb": 0.01}, transfer_metrics=None)
    assert result["verdict"] == "NO_EDGE_IN_SCOPE"


def test_stage58_returns_medium_edge_for_strong_transfer(tmp_path: Path) -> None:
    artifact = tmp_path / "transfer_metrics_real.json"
    artifact.write_text("{}", encoding="utf-8")
    result = assess_transfer_validation(
        stage57_verdict="PASSING_EDGE",
        primary_metrics={"exp_lcb": 0.01},
        transfer_metrics={"exp_lcb": 0.009, "maxDD": 0.18},
        transfer_metric_source_type="real_transfer",
        transfer_artifact_path=str(artifact),
    )
    assert result["verdict"] == "MEDIUM_EDGE"


def test_stage58_preserves_primary_edge_when_transfer_not_acceptable(tmp_path: Path) -> None:
    artifact = tmp_path / "transfer_metrics_real.json"
    artifact.write_text("{}", encoding="utf-8")
    result = assess_transfer_validation(
        stage57_verdict="PASSING_EDGE",
        primary_metrics={"exp_lcb": 0.01},
        transfer_metrics={"exp_lcb": -0.001, "maxDD": 0.30},
        transfer_metric_source_type="real_transfer",
        transfer_artifact_path=str(artifact),
    )
    assert result["verdict"] == "PARTIAL"
    assert result["transfer_acceptable"] is False


def test_stage58_blocks_non_real_transfer_source_even_with_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "transfer_metrics_real.json"
    artifact.write_text("{}", encoding="utf-8")
    result = assess_transfer_validation(
        stage57_verdict="PASSING_EDGE",
        primary_metrics={"exp_lcb": 0.01},
        transfer_metrics={"exp_lcb": 0.02, "maxDD": 0.10},
        transfer_metric_source_type="proxy_only",
        transfer_artifact_path=str(artifact),
    )
    assert result["verdict"] == "PARTIAL"
    assert result["reason"] == "transfer_evidence_not_real"


def test_stage58_propagates_stale_inputs() -> None:
    result = assess_transfer_validation(
        stage57_verdict="STALE_INPUTS",
        primary_metrics={"exp_lcb": 0.01},
        transfer_metrics=None,
    )
    assert result["verdict"] == "STALE_INPUTS"
