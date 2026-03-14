from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def test_stage67_writes_real_validation_artifacts(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    runs = tmp_path / "runs"
    run_id = "r67_test"
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
    (docs / "stage62_summary.json").write_text(
        json.dumps({"stage": "62", "stage28_run_id": run_id}, indent=2),
        encoding="utf-8",
    )

    cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / "run_stage67.py"),
        "--config",
        str(CONFIG_PATH),
        "--runs-dir",
        str(runs),
        "--docs-dir",
        str(docs),
        "--stage28-run-id",
        run_id,
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"run_stage67.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    assert (runs / run_id / "stage67" / "replay_metrics_real.json").exists()
    assert (runs / run_id / "stage67" / "walkforward_metrics_real.json").exists()
    assert (runs / run_id / "stage67" / "split_perturbation_metrics_real.json").exists()
    assert (runs / run_id / "stage57" / "monte_carlo_metrics_real.json").exists()
    assert (runs / run_id / "stage57" / "cross_perturbation_metrics_real.json").exists()


def test_stage67_blocks_frozen_scope_without_resolved_end_ts(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    runs = tmp_path / "runs"
    run_id = "r67_frozen"
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
    (docs / "stage62_summary.json").write_text(
        json.dumps({"stage": "62", "stage28_run_id": run_id}, indent=2),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config.setdefault("reproducibility", {})["frozen_research_mode"] = True
    config["reproducibility"]["require_resolved_end_ts"] = True
    config.setdefault("universe", {})["resolved_end_ts"] = None
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / "run_stage67.py"),
        "--config",
        str(config_path),
        "--runs-dir",
        str(runs),
        "--docs-dir",
        str(docs),
        "--stage28-run-id",
        run_id,
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"run_stage67.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    summary = json.loads((docs / "stage67_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PARTIAL"
    assert summary["runtime_truth_blocked"] is True
    assert summary["runtime_truth_reason"] == "MISSING_RESOLVED_END_TS"
    assert summary["effective_values"]["resolved_end_required_effective"] is True
