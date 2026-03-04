from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REQUIRED_KEYS = {
    "stage",
    "run_id",
    "seed",
    "mode",
    "config_hash",
    "data_hash",
    "window_counts",
    "wf_executed_pct",
    "mc_trigger_pct",
    "top_contextual_edges",
    "policy_metrics",
    "feasibility_summary",
    "verdict",
    "next_bottleneck",
    "summary_hash",
}


def test_stage28_master_summary_schema(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    runs_dir = tmp_path / "runs"
    cmd = [
        sys.executable,
        "scripts/run_stage28.py",
        "--seed",
        "42",
        "--mode",
        "live",
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
    payload = json.loads((docs_dir / "stage28_master_summary.json").read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    assert not missing, f"missing keys: {missing}"
    assert str(payload["stage"]) == "28"
    assert isinstance(payload["policy_metrics"], dict)
    assert "research" in payload["policy_metrics"]
    assert "live" in payload["policy_metrics"]
    assert isinstance(payload["top_contextual_edges"], list)

