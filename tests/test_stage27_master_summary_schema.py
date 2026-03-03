from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def test_stage27_master_summary_schema(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    _write_json(
        docs_dir / "stage27_data_fix_summary.json",
        {
            "coverage_years_by_symbol": {"BTC/USDT": 4.0, "ETH/USDT": 4.0},
            "used_symbols": ["BTC/USDT", "ETH/USDT"],
            "data_snapshot_id": "DATA_FROZEN_v1",
            "data_snapshot_hash": "abc123",
        },
    )
    _write_json(
        docs_dir / "stage27_feasibility_summary.json",
        {
            "top_reject_reasons": [{"reason": "SIZE_TOO_SMALL", "count": 100}],
            "feasibility_rows": [
                {"timeframe": "1h", "recommended_risk_floor": 0.01},
                {"timeframe": "4h", "recommended_risk_floor": 0.02},
            ],
        },
    )
    _write_json(
        docs_dir / "stage27_rerun_summary.json",
        {
            "used_symbols": ["BTC/USDT", "ETH/USDT"],
            "coverage_years_by_symbol": {"BTC/USDT": 4.0, "ETH/USDT": 4.0},
            "data_snapshot_id": "DATA_FROZEN_v1",
            "data_snapshot_hash": "abc123",
            "stages": [
                {"stage": "24", "verdict": "SIZING_ACTIVE_NO_EDGE_CHANGE"},
                {"stage": "25_research", "verdict": "NO_EDGE_IN_RESEARCH"},
                {"stage": "25_live", "verdict": "NO_EDGE_IN_LIVE"},
                {"stage": "26", "verdict": "NO_EDGE"},
            ],
        },
    )
    _write_json(
        docs_dir / "stage26_report_summary.json",
        {
            "verdict": "NO_EDGE",
            "global_baseline_metrics": {"exp_lcb": 0.0, "trade_count": 0.0},
            "conditional_policy_metrics_live": {"exp_lcb": -1.0, "trade_count": 100.0},
            "next_bottleneck": "live_feasibility_constraints",
        },
    )
    _write_json(
        docs_dir / "stage27_research_engine_summary.json",
        {
            "run_id": "",
        },
    )
    _write_json(
        docs_dir / "stage15_9_signal_flow_bottleneck_summary.json",
        {
            "before_after": {"post": {"death_execution": 0.61}},
        },
    )

    report_md = docs_dir / "stage27_master_report.md"
    report_json = docs_dir / "stage27_master_summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage27_master_report.py",
            "--docs-dir",
            str(docs_dir),
            "--report-md",
            str(report_md),
            "--report-json",
            str(report_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    assert report_md.exists()
    assert report_json.exists()
    data = json.loads(report_json.read_text(encoding="utf-8"))
    required = [
        "stage",
        "head_commit",
        "coverage_years_per_symbol",
        "used_symbols",
        "data_snapshot_id",
        "data_snapshot_hash",
        "death_execution_rate",
        "top_reject_reasons",
        "feasibility_min_required_risk_floor",
        "global_baseline_metrics",
        "conditional_policy_metrics_live",
        "contextual_edge_rows",
        "contextual_policy_verdict",
        "stage24_verdict",
        "stage25_research_verdict",
        "stage25_live_verdict",
        "stage26_verdict",
        "final_verdict",
        "next_bottleneck",
        "next_actions",
    ]
    for key in required:
        assert key in data
    assert data["final_verdict"] in {"ROBUST_EDGE", "CONTEXTUAL_EDGE_ONLY", "NO_EDGE", "INSUFFICIENT_DATA"}
    assert data["stage"] == "27"
