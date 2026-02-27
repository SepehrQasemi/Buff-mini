"""Stage-6 compare report generation tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from buffmini.utils.hashing import stable_hash


def _load_module() -> Any:
    module_path = Path("scripts/run_stage6_compare.py").resolve()
    spec = importlib.util.spec_from_file_location("stage6_compare_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_stage6_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary_payload() -> dict[str, Any]:
    resolved_end_ts = "2026-02-27T00:00:00+00:00"
    baseline = {
        "resolved_end_ts": resolved_end_ts,
        "expected_log_growth": 0.0100,
        "return_p05": -0.0020,
        "return_median": 0.0008,
        "return_p95": 0.0021,
        "maxdd_p95": 0.12,
        "p_ruin": None,
        "trades_per_month": 7.5,
        "avg_leverage": 2.0,
        "regime_distribution": {"TREND": 40.0, "RANGE": 45.0, "VOL_EXPANSION": 15.0},
        "execution_drag_sensitivity_delta": -0.01,
    }
    stage6 = {
        "resolved_end_ts": resolved_end_ts,
        "expected_log_growth": 0.0130,
        "return_p05": -0.0010,
        "return_median": 0.0010,
        "return_p95": 0.0024,
        "maxdd_p95": 0.10,
        "p_ruin": None,
        "trades_per_month": 7.5,
        "avg_leverage": 2.2,
        "regime_distribution": {"TREND": 42.0, "RANGE": 43.0, "VOL_EXPANSION": 15.0},
        "execution_drag_sensitivity_delta": -0.008,
    }
    return {
        "baseline_run_id": "baseline_x",
        "stage6_run_id": "stage6_x",
        "source_pipeline_run_id": "pipeline_x",
        "method": "equal",
        "base_leverage": 2.0,
        "seed": 42,
        "resolved_end_ts": resolved_end_ts,
        "window_months": 3,
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "baseline": baseline,
        "stage6": stage6,
    }


def test_stage6_compare_report_files_and_schema(tmp_path: Path) -> None:
    module = _load_module()
    compare_dir = tmp_path / "runs" / "stage6_x" / "stage6_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_payload()

    module._write_compare_report(compare_dir=compare_dir, summary=summary)
    module._write_docs_stage6_report(summary=summary, docs_path=tmp_path / "docs" / "stage6_report.md")

    assert (compare_dir / "stage6_compare_report.md").exists()
    assert (tmp_path / "docs" / "stage6_report.md").exists()

    required_keys = {
        "baseline_run_id",
        "stage6_run_id",
        "resolved_end_ts",
        "baseline",
        "stage6",
    }
    assert required_keys.issubset(summary.keys())
    assert summary["baseline"]["resolved_end_ts"] == summary["stage6"]["resolved_end_ts"]


def test_stage6_compare_summary_hash_is_stable() -> None:
    payload = _summary_payload()
    first = stable_hash(payload, length=20)
    second = stable_hash(json.loads(json.dumps(payload)), length=20)
    assert first == second

