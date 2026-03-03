"""Shared helpers for Stage-26.9 master data audit reporting."""

from __future__ import annotations

from typing import Any


def build_master_summary(
    *,
    raw_payload: dict[str, Any],
    canonical_payload: dict[str, Any],
    derived_payload: dict[str, Any],
    disk_usage: dict[str, float],
    raw_exit_code: int,
    canonical_exit_code: int,
) -> dict[str, Any]:
    raw_rows = list(raw_payload.get("rows", []))
    canonical_rows = list(canonical_payload.get("rows", []))

    coverage_years = {str(row.get("symbol", "")): float(row.get("coverage_years", 0.0)) for row in raw_rows}
    gaps_detected = {str(row.get("symbol", "")): int(dict(row.get("gaps_detected", {})).get("count", 0)) for row in raw_rows}

    canonical_counts: dict[str, dict[str, int]] = {}
    for row in canonical_rows:
        symbol = str(row.get("symbol", ""))
        timeframe = str(row.get("timeframe", ""))
        canonical_counts.setdefault(symbol, {})[timeframe] = int(row.get("rows", 0))

    integrity_pass = bool(
        bool(canonical_payload.get("integrity_pass", False))
        and bool(derived_payload.get("integrity_pass", False))
        and all(float(v) > 0.0 for v in coverage_years.values())
    )

    return {
        "stage": "26.9.4",
        "coverage_years_per_symbol": coverage_years,
        "canonical_candle_counts_per_tf": canonical_counts,
        "derived_tf_supported": list(derived_payload.get("supported", [])),
        "integrity_pass": integrity_pass,
        "gaps_detected": gaps_detected,
        "disk_usage_mb": {
            "raw": float(disk_usage.get("raw", 0.0)),
            "canonical": float(disk_usage.get("canonical", 0.0)),
            "derived": float(disk_usage.get("derived", 0.0)),
            "total": float(disk_usage.get("total", 0.0)),
        },
        "raw_audit_exit_code": int(raw_exit_code),
        "canonical_audit_exit_code": int(canonical_exit_code),
        "derived_sanity": derived_payload,
    }


def render_master_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage-26.9 Data Master Report",
        "",
        f"- integrity_pass: `{bool(summary.get('integrity_pass', False))}`",
        "",
        "## Coverage",
        "",
        "| symbol | coverage_years | gaps_detected |",
        "| --- | ---: | ---: |",
    ]
    for symbol, years in dict(summary.get("coverage_years_per_symbol", {})).items():
        gaps = int(dict(summary.get("gaps_detected", {})).get(symbol, 0))
        lines.append(f"| {symbol} | {float(years):.6f} | {gaps} |")

    lines.extend(
        [
            "",
            "## Disk Usage (MB)",
            "",
            f"- raw: `{float(dict(summary.get('disk_usage_mb', {})).get('raw', 0.0)):.3f}`",
            f"- canonical: `{float(dict(summary.get('disk_usage_mb', {})).get('canonical', 0.0)):.3f}`",
            f"- derived: `{float(dict(summary.get('disk_usage_mb', {})).get('derived', 0.0)):.3f}`",
            f"- total: `{float(dict(summary.get('disk_usage_mb', {})).get('total', 0.0)):.3f}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"
