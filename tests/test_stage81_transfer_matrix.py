from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def test_stage81_summarizes_transfer_matrix(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "stage58_summary.json").write_text(
        json.dumps(
            {
                "stage": "58",
                "primary_symbol": "BTC/USDT",
                "transfer_matrix": [
                    {
                        "transfer_symbol": "ETH/USDT",
                        "classification": "partially_transferable",
                        "diagnostics": ["trigger_rarity"],
                    }
                ],
                "transfer_class_counts": {"partially_transferable": 1},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / "run_stage81.py"),
        "--config",
        str(CONFIG_PATH),
        "--docs-dir",
        str(docs),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"run_stage81.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    summary = json.loads((docs / "stage81_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "SUCCESS"
    assert summary["transfer_matrix_rows"] == 1
    assert summary["failure_diagnostics"]["trigger_rarity"] == 1
