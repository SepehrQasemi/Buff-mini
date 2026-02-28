"""Stage-10.6 report schema tests."""

from __future__ import annotations

import json
from pathlib import Path

from buffmini.stage10.evaluate import build_stage10_6_report_from_runs, validate_stage10_6_summary_schema


def _write_stage10_summary(run_dir: Path, run_id: str, dry_run: bool, stage10_trade_count: float) -> None:
    payload = {
        "stage": "10",
        "run_id": run_id,
        "seed": 42,
        "dry_run": bool(dry_run),
        "config_hash": "cfg123",
        "data_hash": "data123",
        "enabled_signal_families": ["BreakoutRetest", "MA_SlopePullback"],
        "exit_modes": ["fixed_atr", "atr_trailing"],
        "determinism": {"status": "PASS", "signature": "abc123"},
        "baseline_vs_stage10": {
            "baseline": {
                "trade_count": 100.0,
                "profit_factor": 1.1,
                "expectancy": 1.0,
                "max_drawdown": 0.2,
                "pf_adj": 1.05,
                "exp_lcb": 0.8,
            },
            "stage10": {
                "trade_count": float(stage10_trade_count),
                "profit_factor": 1.2,
                "expectancy": 1.1,
                "max_drawdown": 0.19,
                "pf_adj": 1.1,
                "exp_lcb": 0.9,
            },
            "real_data": {"available": not bool(dry_run)},
        },
        "walkforward_v2": {
            "stage10_classification": "UNSTABLE",
            "stage10": {"usable_windows": 3},
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stage10_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_stage10_6_report_schema_generation(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    docs_dir = tmp_path / "docs"

    _write_stage10_summary(runs_dir / "20260228_000001_a_stage10", "20260228_000001_a_stage10", dry_run=True, stage10_trade_count=98.0)
    _write_stage10_summary(runs_dir / "20260228_000002_b_stage10", "20260228_000002_b_stage10", dry_run=True, stage10_trade_count=97.0)
    _write_stage10_summary(runs_dir / "20260228_000003_c_stage10", "20260228_000003_c_stage10", dry_run=False, stage10_trade_count=92.0)
    _write_stage10_summary(runs_dir / "20260228_000004_d_stage10", "20260228_000004_d_stage10", dry_run=False, stage10_trade_count=91.0)

    sandbox_dir = runs_dir / "20260228_000005_x_stage10_sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    (sandbox_dir / "sandbox_summary.json").write_text(
        json.dumps(
            {
                "run_id": "20260228_000005_x_stage10_sandbox",
                "enabled_signals": ["BreakoutRetest", "MA_SlopePullback"],
                "disabled_signals": ["RangeFade"],
                "rank_table_path": str(sandbox_dir / "sandbox_rankings.csv"),
            }
        ),
        encoding="utf-8",
    )

    summary = build_stage10_6_report_from_runs(runs_root=runs_dir, docs_dir=docs_dir, max_drop_pct=10.0)
    validate_stage10_6_summary_schema(summary)

    assert summary["stage"] == "10.6"
    assert (docs_dir / "stage10_6_report.md").exists()
    assert (docs_dir / "stage10_6_report_summary.json").exists()
    assert "dry_run" in summary["comparisons"]
    assert "real_data" in summary["comparisons"]
