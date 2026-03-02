"""Stage-23 unified baseline-vs-enabled audit helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from buffmini.forensics.signal_flow import run_signal_flow_trace
from buffmini.utils.time import utc_now_compact


def run_stage23_audit(
    *,
    config: dict[str, Any],
    seed: int,
    dry_run: bool,
    symbols: list[str],
    timeframes: list[str],
    mode: str,
    stages: list[str],
    families: list[str],
    composers: list[str],
    max_combos: int,
    runs_root: Path,
    data_dir: Path,
    derived_dir: Path,
    docs_dir: Path = Path("docs"),
) -> dict[str, Any]:
    cfg_baseline = deepcopy(config)
    cfg_baseline.setdefault("evaluation", {}).setdefault("stage23", {})["enabled"] = False
    baseline = run_signal_flow_trace(
        config=cfg_baseline,
        seed=int(seed),
        symbols=list(symbols),
        timeframes=list(timeframes),
        mode=str(mode),
        stages=list(stages),
        families=list(families),
        composers=list(composers),
        max_combos=int(max_combos),
        dry_run=bool(dry_run),
        runs_root=runs_root,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )

    cfg_after = deepcopy(config)
    cfg_after.setdefault("evaluation", {}).setdefault("stage23", {})["enabled"] = True
    after = run_signal_flow_trace(
        config=cfg_after,
        seed=int(seed),
        symbols=list(symbols),
        timeframes=list(timeframes),
        mode=str(mode),
        stages=list(stages),
        families=list(families),
        composers=list(composers),
        max_combos=int(max_combos),
        dry_run=bool(dry_run),
        runs_root=runs_root,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )

    baseline_summary = dict(baseline["summary"])
    after_summary = dict(after["summary"])
    baseline_breakdown = _read_json(Path(baseline["trace_dir"]) / "execution_reject_breakdown.json")
    after_breakdown = _read_json(Path(after["trace_dir"]) / "execution_reject_breakdown.json")

    metrics_baseline = _metrics_from_summary(baseline_summary)
    metrics_after = _metrics_from_summary(after_summary)
    death_baseline = _death_from_summary(baseline_summary)
    death_after = _death_from_summary(after_summary)
    top_reasons_baseline = _top_reject_reasons(baseline_breakdown, limit=10)
    top_reasons_after = _top_reject_reasons(after_breakdown, limit=10)

    deltas = {key: float(metrics_after[key] - metrics_baseline[key]) for key in metrics_baseline}
    deltas.update({f"death_{key}": float(death_after[key] - death_baseline[key]) for key in death_baseline})
    biggest_bottleneck = (after_summary.get("top_bottlenecks") or [{}])[0]

    payload = {
        "stage": "23",
        "generated_at": utc_now_compact(),
        "seed": int(seed),
        "baseline": {
            "run_id": str(baseline["run_id"]),
            "trace_dir": str(baseline["trace_dir"]),
            "summary": baseline_summary,
            "execution_reject_breakdown": baseline_breakdown,
            "metrics": metrics_baseline,
            "death_rates": death_baseline,
            "top_reject_reasons": top_reasons_baseline,
        },
        "after": {
            "run_id": str(after["run_id"]),
            "trace_dir": str(after["trace_dir"]),
            "summary": after_summary,
            "execution_reject_breakdown": after_breakdown,
            "metrics": metrics_after,
            "death_rates": death_after,
            "top_reject_reasons": top_reasons_after,
        },
        "deltas": deltas,
        "biggest_remaining_bottleneck": biggest_bottleneck,
        "same_seed": bool(baseline_summary.get("seed") == after_summary.get("seed")),
        "same_data_hash": bool(str(baseline_summary.get("data_hash", "")) == str(after_summary.get("data_hash", ""))),
    }

    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / "stage23_report.md"
    report_json = docs_dir / "stage23_report_summary.json"
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    report_md.write_text(render_stage23_md(payload), encoding="utf-8")
    return {
        "baseline_run_id": baseline["run_id"],
        "after_run_id": after["run_id"],
        "baseline_trace_dir": baseline["trace_dir"],
        "after_trace_dir": after["trace_dir"],
        "report_md": report_md,
        "report_json": report_json,
        "payload": payload,
    }


def render_stage23_md(payload: dict[str, Any]) -> str:
    b = payload["baseline"]
    a = payload["after"]
    d = payload["deltas"]
    bottleneck = payload.get("biggest_remaining_bottleneck", {})
    lines = [
        "# Stage-23 Unified Choke Repair Report",
        "",
        f"- seed: `{payload['seed']}`",
        f"- baseline_run_id: `{b['run_id']}`",
        f"- after_run_id: `{a['run_id']}`",
        f"- same_seed: `{payload['same_seed']}`",
        f"- same_data_hash: `{payload['same_data_hash']}`",
        "",
        "## Baseline vs After",
        "| metric | baseline | after | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("zero_trade_pct", "invalid_pct", "walkforward_executed_true_pct", "mc_trigger_rate"):
        lines.append(
            f"| {key} | {float(b['metrics'][key]):.6f} | {float(a['metrics'][key]):.6f} | {float(d[key]):.6f} |"
        )
    for key in ("context", "orders", "execution"):
        lines.append(
            f"| death_{key} | {float(b['death_rates'][key]):.6f} | {float(a['death_rates'][key]):.6f} | {float(d[f'death_{key}']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Top 10 Execution Reject Reasons (Baseline vs After)",
            "| reason | baseline_count | after_count |",
            "| --- | ---: | ---: |",
        ]
    )
    reason_index = {str(item["reason"]): int(item["count"]) for item in b.get("top_reject_reasons", [])}
    for item in a.get("top_reject_reasons", []):
        reason = str(item["reason"])
        lines.append(f"| {reason} | {int(reason_index.get(reason, 0))} | {int(item['count'])} |")

    lines.extend(
        [
            "",
            "## Evidence Paths",
            f"- baseline trace: `{b['trace_dir']}`",
            f"- after trace: `{a['trace_dir']}`",
            f"- baseline reject breakdown: `{Path(b['trace_dir']) / 'execution_reject_breakdown.json'}`",
            f"- after reject breakdown: `{Path(a['trace_dir']) / 'execution_reject_breakdown.json'}`",
            "",
            "## Biggest Remaining Bottleneck",
            f"- gate: `{bottleneck.get('gate', '')}`",
            f"- death_rate: `{float(bottleneck.get('death_rate', 0.0)):.6f}`",
            "",
            "## Next Actions",
            "- If still choked: address the single largest death-rate gate first before any new signal complexity.",
            "- If WF execution improved: run constrained timeframe sweeps and keep Stage-8/12 validity gates unchanged.",
            "- If MC still low: increase trade density via signal timing quality, not by lowering MC preconditions.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _metrics_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "zero_trade_pct": float(summary.get("zero_trade_pct", 0.0)),
        "invalid_pct": float(summary.get("invalid_pct", 0.0)),
        "walkforward_executed_true_pct": float(summary.get("walkforward_executed_true_pct", 0.0)),
        "mc_trigger_rate": float(summary.get("mc_trigger_rate", 0.0)),
    }


def _death_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    means = dict(summary.get("overall_gate_means", {}) or {})
    return {
        "context": float(means.get("death_context", 0.0)),
        "orders": float(means.get("death_orders", 0.0)),
        "execution": float(means.get("death_execution", 0.0)),
    }


def _top_reject_reasons(payload: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    counts = dict(payload.get("reject_reason_counts", {}) or {})
    rows = [{"reason": str(reason), "count": int(count)} for reason, count in counts.items() if int(count) > 0]
    rows = sorted(rows, key=lambda item: (-item["count"], item["reason"]))
    return rows[: int(limit)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))

