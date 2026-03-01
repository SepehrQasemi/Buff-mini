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
) -> None:
    payload = dict(metrics)
    payload["status"] = str(status)
    payload["warnings"] = [str(item) for item in failures]
    if extras:
        payload.update(dict(extras))
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    lines = [
        f"# {title}",
        "",
        "## 1) What changed",
        f"- status: `{status}`",
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
    if failures:
        lines.extend([f"- {item}" for item in failures])
    else:
        lines.append("- none")
    lines.extend(["", "## 6) Next actions"])
    lines.extend([f"- {item}" for item in next_actions])
    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def summary_hash(payload: dict[str, Any]) -> str:
    return stable_hash(payload, length=16)

