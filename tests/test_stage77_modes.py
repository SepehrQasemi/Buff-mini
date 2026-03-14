from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from buffmini.config import load_config
from buffmini.research.modes import build_mode_context


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def test_stage77_evaluation_mode_blocks_without_pinned_end() -> None:
    cfg = load_config(CONFIG_PATH)
    _, summary = build_mode_context(cfg, requested_mode="evaluation", auto_pin_resolved_end=False)
    assert summary["evaluation_mode"] is True
    assert summary["interpretation_allowed"] is False
    assert "RESOLVED_END_TS_MISSING" in summary["blocked_reasons"]
    assert summary["canonical_status"] == "EVALUATION_BLOCKED"


def test_stage77_exploration_cannot_masquerade_as_evaluation() -> None:
    cfg = load_config(CONFIG_PATH)
    _, summary = build_mode_context(cfg, requested_mode="exploration", auto_pin_resolved_end=False)
    assert summary["evaluation_mode"] is False
    assert summary["interpretation_allowed"] is False
    assert "EXPLORATION_MODE_NOT_INTERPRETABLE_BY_DEFAULT" in summary["blocked_reasons"]
    assert summary["canonical_status"] == "EXPLORATORY"


def test_stage77_runner_writes_artifacts(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / "run_stage77.py"),
        "--config",
        str(CONFIG_PATH),
        "--docs-dir",
        str(docs),
        "--mode",
        "evaluation",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"run_stage77.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    summary = json.loads((docs / "stage77_summary.json").read_text(encoding="utf-8"))
    assert summary["stage"] == "77"
    assert summary["mode"] == "evaluation"
    assert summary["status"] == "PARTIAL"
