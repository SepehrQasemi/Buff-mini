from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_stage28_orchestrator_contract(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    runs_dir = tmp_path / "runs"
    cmd = [
        sys.executable,
        "scripts/run_stage28.py",
        "--seed",
        "42",
        "--mode",
        "research",
        "--dry-run",
        "--budget-small",
        "--symbols",
        "BTC/USDT",
        "--timeframes",
        "1h",
        "--windows",
        "3m",
        "--step-months",
        "3",
        "--docs-dir",
        str(docs_dir),
        "--runs-dir",
        str(runs_dir),
    ]
    done = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr or done.stdout

    report_md = docs_dir / "stage28_master_report.md"
    report_json = docs_dir / "stage28_master_summary.json"
    product_spec = docs_dir / "stage28_product_spec.md"
    assert report_md.exists()
    assert report_json.exists()
    assert product_spec.exists()

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert str(payload.get("stage")) == "28"
    assert int(payload.get("seed", -1)) == 42
    assert isinstance(payload.get("window_counts"), dict)
    assert "policy_metrics" in payload
    assert "verdict" in payload

