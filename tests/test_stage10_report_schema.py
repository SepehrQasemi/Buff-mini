"""Stage-10 summary schema tests."""

from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.stage10.evaluate import run_stage10, validate_stage10_summary_schema


def test_stage10_report_schema_keys(tmp_path: Path) -> None:
    config = load_config(Path("configs/default.yaml"))
    summary = run_stage10(
        config=config,
        seed=42,
        dry_run=True,
        cost_mode="v2",
        walkforward_v2_enabled=False,
        runs_root=tmp_path / "runs",
        docs_dir=tmp_path / "docs",
    )
    validate_stage10_summary_schema(summary)

    assert summary["stage"] == "10"
    assert isinstance(summary["run_id"], str) and summary["run_id"]
    assert isinstance(summary["regimes"]["distribution"], dict)
    assert "baseline_vs_stage10" in summary
