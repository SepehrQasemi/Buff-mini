"""Small helpers for writing deterministic stage artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_stage_artifacts(
    *,
    docs_dir: Path,
    stage: str,
    summary: dict[str, Any],
    report_lines: list[str],
) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / f"stage{stage}_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (docs_dir / f"stage{stage}_report.md").write_text(
        "\n".join(report_lines).strip() + "\n",
        encoding="utf-8",
    )


def markdown_kv(title: str, mapping: dict[str, Any]) -> list[str]:
    lines = [f"## {title}"]
    if not mapping:
        lines.append("- none")
        return lines
    for key, value in sorted(mapping.items()):
        lines.append(f"- {key}: `{value}`")
    return lines


def markdown_rows(title: str, rows: list[dict[str, Any]], *, limit: int | None = None) -> list[str]:
    lines = [f"## {title}"]
    if not rows:
        lines.append("- none")
        return lines
    for row in rows[: limit if limit is not None else len(rows)]:
        lines.append(f"- `{json.dumps(row, sort_keys=True)}`")
    return lines
