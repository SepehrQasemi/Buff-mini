from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_stage26_global_baseline_runner_outputs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    docs_dir = tmp_path / "docs"
    cmd = [
        sys.executable,
        "scripts/run_stage26_global_baseline.py",
        "--dry-run",
        "--seed",
        "42",
        "--symbols",
        "BTC/USDT",
        "--timeframes",
        "1h",
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    csv_files = list(runs_dir.glob("*_stage26_global/stage26/global_baseline_results.csv"))
    json_files = list(runs_dir.glob("*_stage26_global/stage26/global_baseline_results.json"))
    assert csv_files
    assert json_files
    frame = pd.read_csv(csv_files[0])
    assert not frame.empty
    for col in ("symbol", "timeframe", "variant", "rulelet", "trade_count", "exp_lcb", "maxDD"):
        assert col in frame.columns

    comparison_doc = docs_dir / "stage26_global_vs_conditional_comparison.md"
    assert comparison_doc.exists()
