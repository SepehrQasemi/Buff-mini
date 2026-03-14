from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from buffmini.config import load_config
from buffmini.research.synthetic_lab import evaluate_detectability_suite


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def test_stage76_detectability_suite_surfaces_real_signal() -> None:
    cfg = load_config(CONFIG_PATH)
    summary = evaluate_detectability_suite(cfg, seed=42)
    assert summary["bad_control_rejection_rate"] >= 0.66
    assert summary["synthetic_winner_recall"] >= 0.75
    assert summary["signal_detection_rate"] >= 0.60
    assert summary["candidate_classes"].get("promising_but_unproven", 0) >= 1


def test_stage76_runner_writes_artifacts(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / "run_stage76.py"),
        "--config",
        str(CONFIG_PATH),
        "--docs-dir",
        str(docs),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"run_stage76.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    summary = json.loads((docs / "stage76_summary.json").read_text(encoding="utf-8"))
    assert summary["stage"] == "76"
    assert summary["candidate_count"] >= 8
    assert "signal_detection_rate" in summary
