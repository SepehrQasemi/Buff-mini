"""Artifact discovery for Stage-5 ui_bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from buffmini.constants import PROJECT_ROOT, RUNS_DIR


def find_best_report_files(run_dir: Path) -> list[Path]:
    """Find best available markdown reports for a pipeline run."""

    reports: list[Path] = []
    summary = _safe_json(Path(run_dir) / "pipeline_summary.json")

    report_values = (summary.get("reports") or {}) if isinstance(summary, dict) else {}
    for value in report_values.values():
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if path.exists() and path.suffix.lower() == ".md":
            reports.append(path)

    stage_run_ids = [
        summary.get("stage1_run_id"),
        summary.get("stage2_run_id"),
        summary.get("stage3_2_run_id"),
        summary.get("stage3_3_run_id"),
        summary.get("stage4_run_id"),
        summary.get("stage4_sim_run_id"),
    ]
    for run_id in stage_run_ids:
        if not run_id:
            continue
        stage_dir = RUNS_DIR / str(run_id)
        if not stage_dir.exists():
            continue
        for candidate in sorted(stage_dir.glob("*.md")):
            reports.append(candidate.resolve())

    unique: list[Path] = []
    seen: set[str] = set()
    for path in reports:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def find_frontier_selector_tables(run_dir: Path) -> dict[str, str]:
    """Find chart-table sources (frontier/selector) for a pipeline run."""

    summary = _safe_json(Path(run_dir) / "pipeline_summary.json")
    results: dict[str, str] = {}

    stage3_2_run_id = summary.get("stage3_2_run_id") if isinstance(summary, dict) else None
    stage3_3_run_id = summary.get("stage3_3_run_id") if isinstance(summary, dict) else None

    if stage3_2_run_id:
        stage3_2_dir = RUNS_DIR / str(stage3_2_run_id)
        for name in ["leverage_frontier.csv", "frontier_table.csv"]:
            path = stage3_2_dir / name
            if path.exists():
                results["frontier_table"] = str(path.resolve())
                break

    if stage3_3_run_id:
        stage3_3_dir = RUNS_DIR / str(stage3_3_run_id)
        selector = stage3_3_dir / "selector_table.csv"
        feasible = stage3_3_dir / "feasible_only.csv"
        if selector.exists():
            results["selector_table"] = str(selector.resolve())
        if feasible.exists():
            results["selector_feasible_table"] = str(feasible.resolve())

    return results


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
