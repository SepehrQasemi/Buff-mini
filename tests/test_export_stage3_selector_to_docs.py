"""Tests for Stage-3.3 selector docs exporter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_exporter_writes_markdown_and_chosen_line(tmp_path: Path) -> None:
    run_id = "run_x_stage3_3_selector"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    docs_path = tmp_path / "docs" / "stage3_3_selector_summary.md"

    summary = {
        "run_id": run_id,
        "settings": {
            "constraints": {
                "max_p_ruin": 0.01,
                "max_dd_p95": 0.25,
                "min_return_p05": 0.0,
            }
        },
        "method_choices": {
            "equal": {"binding_constraints": ["min_return_p05"]},
            "vol": {"binding_constraints": ["min_return_p05"]},
        },
        "overall_choice": {
            "method": "equal",
            "chosen_leverage": 5.0,
        },
    }
    table = pd.DataFrame(
        [
            {"method": "equal", "leverage": 1.0, "expected_log_growth": 0.1, "return_p05": 0.01, "maxdd_p95": 0.05, "p_ruin": 0.0, "pass_all_constraints": True},
            {"method": "equal", "leverage": 5.0, "expected_log_growth": 0.2, "return_p05": 0.02, "maxdd_p95": 0.15, "p_ruin": 0.0, "pass_all_constraints": True},
            {"method": "vol", "leverage": 1.0, "expected_log_growth": 0.09, "return_p05": 0.01, "maxdd_p95": 0.04, "p_ruin": 0.0, "pass_all_constraints": True},
            {"method": "vol", "leverage": 5.0, "expected_log_growth": 0.18, "return_p05": 0.02, "maxdd_p95": 0.14, "p_ruin": 0.0, "pass_all_constraints": True},
        ]
    )
    (run_dir / "selector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table.to_csv(run_dir / "selector_table.csv", index=False)

    script = Path(__file__).resolve().parents[1] / "scripts" / "export_stage3_selector_to_docs.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-id",
            run_id,
            "--runs-dir",
            str(tmp_path / "runs"),
            "--output",
            str(docs_path),
        ],
        check=True,
    )

    assert docs_path.exists()
    text = docs_path.read_text(encoding="utf-8")
    assert "chosen overall: `equal` @ `5.0x`" in text

