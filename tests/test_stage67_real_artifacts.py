from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def test_stage67_writes_real_validation_artifacts(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    runs = tmp_path / "runs"
    run_id = "r67_test"
    stage62_dir = runs / run_id / "stage62"
    stage62_dir.mkdir(parents=True, exist_ok=True)
    rows = 320
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "tp_before_sl_label": np.where(np.arange(rows) % 2 == 0, 1.0, 0.0),
            "expected_net_after_cost_label": np.sin(np.arange(rows) / 20.0) * 0.001 + 0.0005,
            "realized_label_present": 1,
        }
    )
    frame.to_csv(stage62_dir / "training_dataset_v3.csv", index=False)
    docs.mkdir(parents=True, exist_ok=True)
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
    assert (runs / run_id / "stage67" / "walkforward_metrics_real.json").exists()
    assert (runs / run_id / "stage57" / "monte_carlo_metrics_real.json").exists()
    assert (runs / run_id / "stage57" / "cross_perturbation_metrics_real.json").exists()
