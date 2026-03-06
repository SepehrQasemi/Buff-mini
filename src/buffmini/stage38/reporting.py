"""Stage-38 reporting renderers and schema validation."""

from __future__ import annotations

from typing import Any


def render_stage38_flow_report(payload: dict[str, Any]) -> str:
    """Render end-to-end flow report from trace payload."""

    events = list(payload.get("trace_events", []))
    lines = [
        "# Stage-38 End-to-End Flow Report",
        "",
        "## Run Context",
        f"- stage28_run_id: `{payload.get('run_id', '')}`",
        f"- stage28_dir: `{payload.get('stage28_dir', '')}`",
        f"- trace_hash: `{payload.get('trace_hash', '')}`",
        "",
        "## Entrypoints",
    ]
    for item in payload.get("entrypoints", []):
        lines.append(f"- {item.get('name', '')}:")
        flow = list(item.get("flow", []))
        if flow:
            lines.append(f"  {' -> '.join(str(v) for v in flow)}")
        else:
            lines.append("  (none)")

    lines.extend(
        [
            "",
            "## Execution Trace",
            "| order | stage | component | action | input_rows | output_rows | key_details |",
            "| ---: | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for event in events:
        details = dict(event.get("details", {}))
        compact = ", ".join([f"{k}={details[k]}" for k in sorted(details.keys()) if k in {"coverage_gate_status", "policy_context_count", "policy_candidate_count", "candidate_rows_active", "net_score_nonzero_rows", "final_signal_nonzero_rows", "shadow_reject_rows", "verdict", "next_bottleneck"}])
        lines.append(
            "| {order} | {stage} | {component} | {action} | {i} | {o} | {d} |".format(
                order=int(event.get("order", 0)),
                stage=str(event.get("stage", "")),
                component=str(event.get("component", "")),
                action=str(event.get("action", "")),
                i=int(event.get("input_rows", 0)),
                o=int(event.get("output_rows", 0)),
                d=compact or "-",
            )
        )

    counts = dict(payload.get("artifact_row_counts", {}))
    lines.extend(["", "## Artifact Row Counts"])
    for key in sorted(counts.keys()):
        lines.append(f"- {key}: `{int(counts[key])}`")

    lines.extend(
        [
            "",
            "## Suspicious Branch Checks",
            f"- active_candidates_vs_final_signal: `{_fmt_bool(_suspicious_active_without_signal(payload))}`",
            f"- nonzero_net_required_for_signal: `{_fmt_bool(_net_signal_coherence(payload))}`",
            "",
            "## Conclusion",
            "- This report is generated from runtime artifacts (`summary.json`, funnel CSVs, policy trace, rejects).",
            "- Any count mismatch between activation hunt and engine should be diagnosed via Stage-38 logic audit table.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render_stage38_logic_audit_report(payload: dict[str, Any]) -> str:
    """Render Stage-38 contradiction and enforcement audit report."""

    table = dict(payload.get("lineage_table", {}))
    oi = dict(payload.get("oi_usage", {}))
    learning = dict(payload.get("self_learning", {}))
    lines = [
        "# Stage-38 Logic Audit Report",
        "",
        "## Hunt vs Engine Lineage",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in (
        "raw_signal_count",
        "legacy_raw_signal_count",
        "post_threshold_count",
        "post_cost_gate_count",
        "post_feasibility_count",
        "composer_signal_count",
        "engine_raw_signal_count",
        "final_trade_count",
    ):
        lines.append(f"| {key} | {float(table.get(key, 0.0)):.6f} |")
    root_cause = str(payload.get("root_cause", "")).strip()
    fix_summary = str(payload.get("fix_summary", "")).strip()
    before_after = list(payload.get("before_after", []))
    if before_after:
        lines.extend(["", "## Before vs After Evidence", "| metric | before | after |", "| --- | ---: | ---: |"])
        for row in before_after:
            lines.append(
                "| {metric} | {before:.6f} | {after:.6f} |".format(
                    metric=str(row.get("metric", "")),
                    before=float(row.get("before", 0.0)),
                    after=float(row.get("after", 0.0)),
                )
            )
    lines.extend(
        [
            "",
            "## Collapse Point",
            f"- collapse_reason: `{payload.get('collapse_reason', '')}`",
            f"- contradiction_fixed: `{bool(payload.get('contradiction_fixed', False))}`",
            f"- root_cause: `{root_cause}`",
            f"- fix_summary: `{fix_summary}`",
            "",
            "## OI Short-Only Enforcement",
            f"- short_only_enabled: `{bool(oi.get('short_only_enabled', False))}`",
            f"- timeframe: `{oi.get('timeframe', '')}`",
            f"- timeframe_allowed: `{bool(oi.get('timeframe_allowed', False))}`",
            f"- oi_active_runtime: `{bool(oi.get('oi_active_runtime', False))}`",
            f"- oi_non_null_rows: `{int(oi.get('oi_non_null_rows', 0))}`",
            "",
            "## Self-Learning Registry",
            f"- registry_path: `{learning.get('registry_path', '')}`",
            f"- registry_rows: `{int(learning.get('registry_rows', 0))}`",
            f"- elites_count: `{int(learning.get('elites_count', 0))}`",
            f"- dead_family_count: `{int(learning.get('dead_family_count', 0))}`",
            f"- failure_motif_tags_non_empty: `{bool(learning.get('failure_motif_tags_non_empty', False))}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def validate_stage38_master_summary(payload: dict[str, Any]) -> None:
    """Validate Stage-38 master summary schema."""

    required = {
        "stage",
        "seed",
        "stage28_run_id",
        "trace_hash",
        "lineage_table",
        "collapse_reason",
        "contradiction_fixed",
        "oi_usage",
        "self_learning",
        "verdict",
        "biggest_remaining_bottleneck",
        "next_action",
    }
    missing = sorted(required.difference(set(payload.keys())))
    if missing:
        raise ValueError(f"Missing Stage-38 summary keys: {missing}")
    if str(payload.get("stage", "")).strip() != "38.6":
        raise ValueError("stage must be '38.6'")
    lineage = payload.get("lineage_table", {})
    if not isinstance(lineage, dict):
        raise ValueError("lineage_table must be object")
    for key in (
        "raw_signal_count",
        "legacy_raw_signal_count",
        "post_threshold_count",
        "post_cost_gate_count",
        "post_feasibility_count",
        "composer_signal_count",
        "engine_raw_signal_count",
        "final_trade_count",
    ):
        if key not in lineage:
            raise ValueError(f"lineage_table.{key} missing")
    if not isinstance(payload.get("oi_usage", {}), dict):
        raise ValueError("oi_usage must be object")
    if not isinstance(payload.get("self_learning", {}), dict):
        raise ValueError("self_learning must be object")


def _suspicious_active_without_signal(payload: dict[str, Any]) -> bool:
    events = list(payload.get("trace_events", []))
    composer = next((event for event in events if str(event.get("stage", "")) == "composer"), {})
    details = dict(composer.get("details", {}))
    active = int(details.get("candidate_rows_active", 0))
    final_nonzero = int(details.get("final_signal_nonzero_rows", 0))
    return bool(active > 0 and final_nonzero <= 0)


def _net_signal_coherence(payload: dict[str, Any]) -> bool:
    events = list(payload.get("trace_events", []))
    composer = next((event for event in events if str(event.get("stage", "")) == "composer"), {})
    details = dict(composer.get("details", {}))
    net_nonzero = int(details.get("net_score_nonzero_rows", 0))
    final_nonzero = int(details.get("final_signal_nonzero_rows", 0))
    return bool((final_nonzero <= net_nonzero) and (net_nonzero >= 0))


def _fmt_bool(value: bool) -> str:
    return "true" if bool(value) else "false"
