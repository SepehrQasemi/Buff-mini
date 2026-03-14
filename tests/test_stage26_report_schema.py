from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_stage26_report_summary_schema(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    docs_dir = tmp_path / "docs"
    data_dir = tmp_path / "data_raw"
    derived_dir = tmp_path / "data_derived"
    cmd = [
        sys.executable,
        "scripts/run_stage26.py",
        "--seed",
        "42",
        "--dry-run",
        "--symbols",
        "BTC/USDT",
        "--timeframes",
        "1h",
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
        "--data-dir",
        str(data_dir),
        "--derived-dir",
        str(derived_dir),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    summary_path = docs_dir / "stage26_report_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {
        "head_commit",
        "data_coverage_years_by_symbol",
        "timeframes_tested",
        "contexts",
        "top_rulelets_per_context",
        "conditional_policy_metrics_research",
        "conditional_policy_metrics_live",
        "shadow_live_reject_rate",
        "global_baseline_metrics",
        "comparison_delta",
        "verdict",
        "next_bottleneck",
    }
    missing = sorted(required.difference(payload.keys()))
    assert not missing, f"Missing keys: {missing}"
    assert payload["coverage_gate_status"] == "INSUFFICIENT_DATA"
    assert payload["coverage_gate_can_run"] is False
    assert payload["verdict"] == "INSUFFICIENT_DATA"
    assert any("dry_run_coverage_gate_override" in warning for warning in payload.get("warnings", []))
