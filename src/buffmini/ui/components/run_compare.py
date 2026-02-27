"""Run compare helpers using standardized ui_bundle artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_ui_bundle_summary(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Load ui_bundle/summary_ui.json for one run."""

    warnings: list[str] = []
    path = Path(run_dir) / "ui_bundle" / "summary_ui.json"
    if not path.exists():
        warnings.append(f"missing {path}")
        return {}, warnings
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"failed parsing {path}: {exc}")
        return {}, warnings
    return payload if isinstance(payload, dict) else {}, warnings


def load_ui_bundle_curve(run_dir: Path, filename: str) -> tuple[pd.DataFrame, list[str]]:
    """Load ui_bundle curve file safely."""

    warnings: list[str] = []
    path = Path(run_dir) / "ui_bundle" / filename
    if not path.exists():
        warnings.append(f"missing {path}")
        return pd.DataFrame(), warnings
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"failed reading {path}: {exc}")
        return pd.DataFrame(), warnings
    return frame, warnings


def load_ui_bundle_charts_index(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Load ui_bundle/charts_index.json."""

    warnings: list[str] = []
    path = Path(run_dir) / "ui_bundle" / "charts_index.json"
    if not path.exists():
        warnings.append(f"missing {path}")
        return {}, warnings
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"failed parsing {path}: {exc}")
        return {}, warnings
    return payload if isinstance(payload, dict) else {}, warnings


def resolve_run_metrics(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Resolve compare metrics for one run from ui_bundle summary and selector row."""

    summary, warnings = load_ui_bundle_summary(run_dir)
    if not summary:
        return {}, warnings

    metrics = {
        "expected_log_growth": _safe_float((summary.get("key_metrics") or {}).get("expected_log_growth")),
        "pf": _safe_float((summary.get("key_metrics") or {}).get("pf")),
        "maxdd_p95": _safe_float((summary.get("key_metrics") or {}).get("maxdd")),
        "p_ruin": _safe_float((summary.get("key_metrics") or {}).get("p_ruin")),
        "return_p05": None,
        "return_median": None,
        "return_p95": None,
        "chosen_method": summary.get("chosen_method"),
        "chosen_leverage": summary.get("chosen_leverage"),
        "execution_mode": summary.get("execution_mode"),
    }

    charts_index, chart_warnings = load_ui_bundle_charts_index(run_dir)
    warnings.extend(chart_warnings)
    selector_path = charts_index.get("selector_table") if isinstance(charts_index, dict) else None
    if selector_path:
        try:
            table = pd.read_csv(Path(selector_path))
        except Exception as exc:
            warnings.append(f"failed reading selector table {selector_path}: {exc}")
        else:
            method = str(summary.get("chosen_method", ""))
            leverage = _safe_float(summary.get("chosen_leverage"))
            if method and leverage is not None and not table.empty and {"method", "leverage"}.issubset(table.columns):
                filtered = table[table["method"].astype(str) == method].copy()
                if not filtered.empty:
                    filtered["lev_dist"] = (pd.to_numeric(filtered["leverage"], errors="coerce") - leverage).abs()
                    row = filtered.sort_values("lev_dist").iloc[0]
                    metrics["return_p05"] = _safe_float(row.get("return_p05"))
                    metrics["return_median"] = _safe_float(row.get("return_median"))
                    metrics["return_p95"] = _safe_float(row.get("return_p95"))
                    metrics["maxdd_p95"] = _safe_float(row.get("maxdd_p95"))
                    metrics["p_ruin"] = _safe_float(row.get("p_ruin"))
                    metrics["expected_log_growth"] = _safe_float(row.get("expected_log_growth"))
    return metrics, warnings


def build_comparison_table(run_a_dir: Path, run_b_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Build side-by-side metrics table for two runs."""

    metrics_a, warnings_a = resolve_run_metrics(run_a_dir)
    metrics_b, warnings_b = resolve_run_metrics(run_b_dir)
    warnings = warnings_a + warnings_b

    metric_order = [
        "expected_log_growth",
        "pf",
        "maxdd_p95",
        "p_ruin",
        "return_p05",
        "return_median",
        "return_p95",
    ]

    rows: list[dict[str, Any]] = []
    for key in metric_order:
        rows.append(
            {
                "metric": key,
                "run_a": metrics_a.get(key),
                "run_b": metrics_b.get(key),
            }
        )

    return pd.DataFrame(rows), warnings


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None
