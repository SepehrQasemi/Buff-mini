"""Artifact-driven run index for Stage-5 UI pages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from buffmini.constants import RUNS_DIR


def scan_runs(runs_dir: Path = RUNS_DIR) -> list[dict[str, Any]]:
    """Scan runs directory and return normalized run records."""

    records: list[dict[str, Any]] = []
    base = Path(runs_dir)
    if not base.exists():
        return records

    for path in sorted(base.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        run_id = path.name
        if run_id.startswith("_"):
            continue
        progress = _safe_json(path / "progress.json")
        pipeline = _safe_json(path / "pipeline_summary.json")
        status = (
            str(progress.get("status"))
            if progress
            else str(pipeline.get("status")) if pipeline else "unknown"
        )
        stage = (
            str(progress.get("stage"))
            if progress
            else str(pipeline.get("current_stage", "")) if pipeline else ""
        )
        records.append(
            {
                "run_id": run_id,
                "path": str(path),
                "status": status,
                "stage": stage,
                "updated_at": path.stat().st_mtime,
                "progress": progress,
                "pipeline_summary": pipeline,
            }
        )
    return records


def latest_completed_pipeline(runs_dir: Path = RUNS_DIR) -> dict[str, Any] | None:
    """Return latest completed pipeline run record."""

    for item in scan_runs(runs_dir):
        summary = item.get("pipeline_summary") or {}
        if summary and str(summary.get("status")) == "success":
            return item
    return None


def latest_run_id(runs_dir: Path = RUNS_DIR) -> str:
    """Return most recent run id or empty string."""

    records = scan_runs(runs_dir)
    return str(records[0]["run_id"]) if records else ""


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

