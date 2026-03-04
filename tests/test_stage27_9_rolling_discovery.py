from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_stage27_9_rolling_discovery_contract(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    docs_dir = tmp_path / "docs"
    data_dir = tmp_path / "data"
    derived_dir = tmp_path / "derived"
    runs_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_stage27_9_rolling_discovery.py",
        "--dry-run",
        "--seed",
        "42",
        "--symbols",
        "BTC/USDT",
        "--timeframes",
        "1h",
        "--windows",
        "3m",
        "--step-size",
        "1m",
        "--max-windows",
        "1",
        "--allow-insufficient-data",
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
        "--data-dir",
        str(data_dir),
        "--derived-dir",
        str(derived_dir),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "run_id:" in result.stdout

    summary_path = docs_dir / "stage27_9_rolling_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload.get("stage") == "27.9.4"
    assert isinstance(payload.get("window_counts"), dict)
    assert isinstance(payload.get("rows"), int)

    run_dirs = sorted([path for path in runs_dir.glob("*_stage27_9_roll") if path.is_dir()])
    assert run_dirs
    rolling_csv = run_dirs[-1] / "stage27_9" / "rolling_results.csv"
    assert rolling_csv.exists()
