"""Shared report helpers for Stage-15..22."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from buffmini.utils.hashing import stable_hash


def write_report_pair(
    *,
    report_md: Path,
    report_json: Path,
    title: str,
    how_to_run: list[str],
    metrics: dict[str, Any],
    status: str,
    failures: list[str],
    next_actions: list[str],
    extras: dict[str, Any] | None = None,
    stage_type: str = "trading",
    expect_walkforward: bool = False,
    expect_mc: bool = False,
    trade_count_key: str = "trade_count",
) -> None:
    final_status, final_failures = truthful_stage_status(
        metrics=metrics,
        status=status,
        failures=failures,
        stage_type=stage_type,
        expect_walkforward=expect_walkforward,
        expect_mc=expect_mc,
        trade_count_key=trade_count_key,
    )
    payload = dict(metrics)
    payload["status"] = str(final_status)
    payload["warnings"] = [str(item) for item in final_failures]
    if extras:
        payload.update(dict(extras))
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    lines = [
        f"# {title}",
        "",
        "## 1) What changed",
        f"- status: `{final_status}`",
        "",
        "## 2) How to run (dry-run + real)",
    ]
    lines.extend([f"- {line}" for line in how_to_run])
    lines.extend(
        [
            "",
            "## 3) Validation gates & results",
        ]
    )
    for key, value in metrics.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## 4) Key metrics tables", "- See JSON summary for full machine-readable values."])
    lines.extend(["", "## 5) Failures + reasons"])
    if final_failures:
        lines.extend([f"- {item}" for item in final_failures])
    else:
        lines.append("- none")
    lines.extend(["", "## 6) Next actions"])
    lines.extend([f"- {item}" for item in next_actions])
    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def truthful_stage_status(
    *,
    metrics: dict[str, Any],
    status: str,
    failures: list[str],
    stage_type: str = "trading",
    expect_walkforward: bool = False,
    expect_mc: bool = False,
    trade_count_key: str = "trade_count",
) -> tuple[str, list[str]]:
    """Enforce truthful PASS/FAIL with deterministic rule set."""

    out_status = str(status)
    out_failures = [str(item) for item in failures]
    if str(stage_type) == "trading":
        trade_count = _to_float(metrics.get(trade_count_key, 0.0))
        if trade_count <= 0.0:
            out_failures.append("truth:no_trades_executed")
    if bool(expect_walkforward):
        wf = _to_float(metrics.get("walkforward_executed_true_pct", 0.0))
        if wf <= 0.0:
            out_failures.append("truth:walkforward_not_executed")
    if bool(expect_mc):
        mc = _to_float(metrics.get("mc_trigger_rate", 0.0))
        if mc <= 0.0:
            out_failures.append("truth:mc_not_triggered")
    # Deterministic de-dup while preserving insertion order.
    seen: set[str] = set()
    dedup = []
    for item in out_failures:
        if item not in seen:
            seen.add(item)
            dedup.append(item)
    if dedup:
        out_status = "FAILED"
    return out_status, dedup


def summary_hash(payload: dict[str, Any]) -> str:
    return stable_hash(payload, length=16)


def _to_float(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return 0.0
    return num
