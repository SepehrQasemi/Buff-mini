"""Robust artifact loaders for Stage-5 UI (never crash on missing files)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import PROJECT_ROOT


def load_stage3_2_artifacts(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Load Stage-3.2 frontier artifacts with graceful fallback."""

    warnings: list[str] = []
    payload: dict[str, Any] = {}
    summary_candidates = [run_dir / "stage3_2_summary.json", run_dir / "frontier_summary.json"]
    table_candidates = [run_dir / "leverage_frontier.csv", run_dir / "frontier_table.csv"]
    payload["summary"] = _first_json(summary_candidates, warnings, "stage3_2_summary")
    payload["table"] = _first_csv(table_candidates, warnings, "stage3_2_table")
    return payload, warnings


def load_stage3_3_artifacts(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Load Stage-3.3 selector artifacts."""

    warnings: list[str] = []
    payload: dict[str, Any] = {}
    payload["summary"] = _first_json([run_dir / "selector_summary.json"], warnings, "selector_summary")
    payload["table"] = _first_csv([run_dir / "selector_table.csv"], warnings, "selector_table")
    payload["feasible"] = _first_csv([run_dir / "feasible_only.csv"], warnings, "feasible_only")
    return payload, warnings


def load_stage4_artifacts(run_dir: Path, project_root: Path = PROJECT_ROOT) -> tuple[dict[str, Any], list[str]]:
    """Load Stage-4 docs/simulation artifacts."""

    warnings: list[str] = []
    docs_dir = Path(project_root) / "docs"
    trading_spec_path = _first_existing(
        [
            run_dir / "spec" / "trading_spec.md",
            run_dir / "trading_spec.md",
            docs_dir / "trading_spec.md",
        ]
    )
    checklist_path = _first_existing(
        [
            run_dir / "spec" / "paper_trading_checklist.md",
            run_dir / "paper_trading_checklist.md",
            docs_dir / "paper_trading_checklist.md",
        ]
    )
    payload: dict[str, Any] = {
        "trading_spec": _safe_text(trading_spec_path, warnings, "stage4/trading_spec.md"),
        "paper_checklist": _safe_text(checklist_path, warnings, "stage4/paper_trading_checklist.md"),
        "execution_metrics": _first_json([run_dir / "execution_metrics.json"], warnings, "execution_metrics"),
        "exposure_timeseries": _first_csv([run_dir / "exposure_timeseries.csv"], warnings, "exposure_timeseries"),
        "orders": _first_csv([run_dir / "orders.csv"], warnings, "orders"),
        "killswitch_events": _first_csv([run_dir / "killswitch_events.csv"], warnings, "killswitch_events"),
    }
    return payload, warnings


def load_generic_curves(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Load generic equity/drawdown curves if present."""

    warnings: list[str] = []
    payload: dict[str, Any] = {
        "equity_curve": _first_csv([run_dir / "equity_curve.csv"], warnings, "equity_curve"),
        "drawdown_curve": _first_csv([run_dir / "drawdown_curve.csv"], warnings, "drawdown_curve"),
    }
    return payload, warnings


def load_pipeline_summary(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Load pipeline summary and progress metadata."""

    warnings: list[str] = []
    payload = {
        "pipeline_summary": _first_json([run_dir / "pipeline_summary.json"], warnings, "pipeline_summary"),
        "progress": _first_json([run_dir / "progress.json"], warnings, "progress"),
    }
    return payload, warnings


def _first_json(candidates: list[Path], warnings: list[str], label: str) -> dict[str, Any]:
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                warnings.append(f"{label}: failed to parse {path.name}: {exc}")
                return {}
    warnings.append(f"{label}: missing")
    return {}


def _first_csv(candidates: list[Path], warnings: list[str], label: str) -> pd.DataFrame:
    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception as exc:
                warnings.append(f"{label}: failed to parse {path.name}: {exc}")
                return pd.DataFrame()
    warnings.append(f"{label}: missing")
    return pd.DataFrame()


def _safe_text(path: Path, warnings: list[str], label: str) -> str:
    if not path.exists():
        warnings.append(f"{label}: missing")
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        warnings.append(f"{label}: failed to read: {exc}")
        return ""


def _first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]
