from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_stage27_rolling_discovery_contract(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    runs_dir = tmp_path / "runs"
    cmd = [
        sys.executable,
        "scripts/run_stage27_rolling_discovery.py",
        "--seed",
        "42",
        "--dry-run",
        "--symbols",
        "BTC/USDT",
        "--timeframes",
        "1h",
        "--windows",
        "3m",
        "--step",
        "1m",
        "--docs-dir",
        str(docs_dir),
        "--runs-dir",
        str(runs_dir),
        "--allow-insufficient-data",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr

    summary_path = docs_dir / "stage27_research_engine_summary.json"
    report_path = docs_dir / "stage27_research_engine_report.md"
    assert summary_path.exists()
    assert report_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in (
        "run_id",
        "used_symbols",
        "semantic_hash_equal",
        "feature_cache_hit_rate_estimate",
        "runtime_seconds",
    ):
        assert key in payload
    run_id = str(payload["run_id"])
    rolling_json = runs_dir / run_id / "stage27" / "rolling_results.json"
    rolling_csv = runs_dir / run_id / "stage27" / "rolling_results.csv"
    assert rolling_json.exists()
    assert rolling_csv.exists()

