from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from buffmini.stage58 import assess_transfer_validation

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


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
        transfer_metrics={"trade_count": 24, "exp_lcb": 0.009, "maxDD": 0.18},
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


def test_stage58_writes_real_transfer_artifact(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    runs = tmp_path / "runs"
    run_id = "r58_test"
    stage52_dir = runs / run_id / "stage52"
    stage52_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "candidate_id": "s52_1",
                "source_candidate_id": "s47_1",
                "family": "structure_pullback_continuation",
                "timeframe": "1h",
                "beam_score": 0.9,
                "cost_edge_proxy": 0.001,
                "rr_model": "{'first_target_rr': 1.8}",
                "geometry": "{'expected_hold_bars': 12, 'stop_distance_pct': 0.005, 'first_target_pct': 0.01, 'stretch_target_pct': 0.015, 'entry_zone': {'low': 0.99, 'high': 1.01}}",
                "eligible_for_replay": True,
            }
        ]
    ).to_csv(stage52_dir / "setup_candidates_v2.csv", index=False)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "stage52_summary.json").write_text(
        json.dumps({"stage": "52", "stage28_run_id": run_id, "representative_candidate_id": "s52_1"}, indent=2),
        encoding="utf-8",
    )
    (docs / "stage57_summary.json").write_text(
        json.dumps({"stage": "57", "verdict": "PASSING_EDGE", "replay_gate": {"exp_lcb": 0.01}}, indent=2),
        encoding="utf-8",
    )

    cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / "run_stage58.py"),
        "--config",
        str(CONFIG_PATH),
        "--runs-dir",
        str(runs),
        "--docs-dir",
        str(docs),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"run_stage58.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    assert (runs / run_id / "stage58" / "transfer_metrics_real.json").exists()
    summary = json.loads((docs / "stage58_summary.json").read_text(encoding="utf-8"))
    assert summary["transfer_artifact_exists"] is True
    assert summary["evidence_quality"] in {"artifact_backed_real", "real_but_blocked"}
