from __future__ import annotations

import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.stage11.evaluate import run_stage11, validate_stage11_summary_schema


def test_stage11_report_schema_and_docs_output(tmp_path) -> None:
    config = load_config(Path("configs/default.yaml"))
    config["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 600
    config["evaluation"]["stage11"]["enabled"] = True
    config["evaluation"]["stage11"]["hooks"]["confirm"]["enabled"] = False
    config["evaluation"]["stage11"]["hooks"]["exit"]["enabled"] = False

    docs_dir = tmp_path / "docs"
    runs_dir = tmp_path / "runs"
    summary = run_stage11(
        config=config,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframe="1h",
        cost_mode="v2",
        walkforward_v2_enabled=False,
        runs_root=runs_dir,
        docs_dir=docs_dir,
        write_docs=True,
    )
    validate_stage11_summary_schema(summary)
    report_json_path = docs_dir / "stage11_report_summary.json"
    report_md_path = docs_dir / "stage11_report.md"
    assert report_json_path.exists()
    assert report_md_path.exists()
    payload = json.loads(report_json_path.read_text(encoding="utf-8"))
    validate_stage11_summary_schema(payload)
    assert payload["run_id"] == summary["run_id"]

