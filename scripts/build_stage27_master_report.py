"""Build Stage-27 master report from stage summaries."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.stage27.edge_verdict import contextual_edge_verdict
from buffmini.utils.time import utc_now_compact


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8-sig")))


def _git_head() -> str:
    head = Path(".git/HEAD")
    if not head.exists():
        return ""
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = Path(".git") / text.split(" ", 1)[1].strip()
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-27 master report")
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--report-md", type=Path, default=Path("docs/stage27_master_report.md"))
    parser.add_argument("--report-json", type=Path, default=Path("docs/stage27_master_summary.json"))
    return parser.parse_args()


def _render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-27 Master Report",
        "",
        f"- generated_at: `{payload.get('generated_at', '')}`",
        f"- head_commit: `{payload.get('head_commit', '')}`",
        "",
        "## Data Status",
        f"- coverage_years_per_symbol: `{payload.get('coverage_years_per_symbol', {})}`",
        f"- used_symbols: `{payload.get('used_symbols', [])}`",
        f"- data_snapshot_id: `{payload.get('data_snapshot_id', '')}`",
        f"- data_snapshot_hash: `{payload.get('data_snapshot_hash', '')}`",
        "",
        "## Execution Health",
        f"- death_execution_rate: `{float(payload.get('death_execution_rate', 0.0)):.6f}`",
        f"- top_reject_reasons: `{payload.get('top_reject_reasons', [])}`",
        f"- feasibility_min_required_risk_floor: `{payload.get('feasibility_min_required_risk_floor', {})}`",
        "",
        "## Research Results",
        f"- global_baseline_metrics: `{payload.get('global_baseline_metrics', {})}`",
        f"- conditional_policy_metrics_live: `{payload.get('conditional_policy_metrics_live', {})}`",
        f"- stage24_verdict: `{payload.get('stage24_verdict', '')}`",
        f"- stage25_research_verdict: `{payload.get('stage25_research_verdict', '')}`",
        f"- stage25_live_verdict: `{payload.get('stage25_live_verdict', '')}`",
        f"- stage26_verdict: `{payload.get('stage26_verdict', '')}`",
        f"- contextual_edge_rows: `{len(payload.get('contextual_edge_rows', []))}`",
        f"- contextual_policy_verdict: `{payload.get('contextual_policy_verdict', '')}`",
        "",
        "## Final Verdict",
        f"- `{payload.get('final_verdict', '')}`",
        f"- next_bottleneck: `{payload.get('next_bottleneck', '')}`",
        "",
        "## Next Actions",
    ]
    for item in payload.get("next_actions", []):
        lines.append(f"- {item}")
    return "\n".join(lines).strip() + "\n"


def _extract_death_execution_rate(trace: dict[str, Any]) -> float:
    before_after = dict(trace.get("before_after", {}))
    post = dict(before_after.get("post", {}))
    direct = post.get("death_execution")
    if direct is not None:
        return float(direct)
    gate_means = dict(post.get("overall_gate_means", {}))
    if "death_execution" in gate_means:
        return float(gate_means["death_execution"])
    top = list(post.get("top_bottlenecks", []))
    for row in top:
        if str(row.get("gate", "")) == "death_execution":
            return float(row.get("death_rate", 0.0))
    return 0.0


def build_stage27_master_report(
    docs_dir: Path,
    report_md: Path,
    report_json: Path,
) -> dict[str, Any]:
    data_fix = _read_json(docs_dir / "stage27_data_fix_summary.json")
    feasibility = _read_json(docs_dir / "stage27_feasibility_summary.json")
    rerun = _read_json(docs_dir / "stage27_rerun_summary.json")
    stage26 = _read_json(docs_dir / "stage26_report_summary.json")
    rolling = _read_json(docs_dir / "stage27_research_engine_summary.json")
    trace = _read_json(docs_dir / "stage15_9_signal_flow_bottleneck_summary.json")

    rolling_rows = pd.DataFrame()
    run_id = str(rolling.get("run_id", ""))
    if run_id:
        csv_path = Path("runs") / run_id / "stage27" / "rolling_results.csv"
        if csv_path.exists():
            rolling_rows = pd.read_csv(csv_path)
    context_verdict = contextual_edge_verdict(rolling_rows)

    final_verdict = "NO_EDGE"
    if not rerun:
        final_verdict = "INSUFFICIENT_DATA"
    elif bool(context_verdict.get("has_contextual_edge", False)):
        final_verdict = "CONTEXTUAL_EDGE_ONLY"
    if str(stage26.get("verdict", "")) == "INSUFFICIENT_DATA":
        final_verdict = "INSUFFICIENT_DATA"
    elif str(stage26.get("verdict", "")) == "ROBUST_EDGE":
        final_verdict = "ROBUST_EDGE"

    top_reasons = list(feasibility.get("top_reject_reasons", []))
    if not top_reasons:
        shadow = dict(stage26.get("shadow_live_top_reasons", {}))
        top_reasons = [
            {"reason": str(reason), "count": int(count)}
            for reason, count in sorted(shadow.items(), key=lambda item: int(item[1]), reverse=True)[:5]
        ]
    feasibility_rows = list(feasibility.get("feasibility_rows", []))
    risk_floor_by_tf: dict[str, float] = {}
    for row in feasibility_rows:
        tf = str(row.get("timeframe", ""))
        value = float(row.get("recommended_risk_floor", 0.0))
        current = risk_floor_by_tf.get(tf)
        if current is None or value > current:
            risk_floor_by_tf[tf] = value

    next_bottleneck = str(stage26.get("next_bottleneck", "")) or str((rerun.get("stages", [{}])[-1] or {}).get("verdict", ""))
    if top_reasons:
        next_bottleneck = f"execution_feasibility:{top_reasons[0].get('reason', '')}"

    stage24_row = next((row for row in rerun.get("stages", []) if str(row.get("stage", "")) == "24"), {})
    stage25_research_row = next((row for row in rerun.get("stages", []) if str(row.get("stage", "")) == "25_research"), {})
    stage25_live_row = next((row for row in rerun.get("stages", []) if str(row.get("stage", "")) == "25_live"), {})
    stage26_row = next((row for row in rerun.get("stages", []) if str(row.get("stage", "")) == "26"), {})

    payload = {
        "stage": "27",
        "generated_at": utc_now_compact(),
        "head_commit": _git_head(),
        "coverage_years_per_symbol": dict(data_fix.get("coverage_years_by_symbol", rerun.get("coverage_years_by_symbol", {}))),
        "used_symbols": list(rerun.get("used_symbols", data_fix.get("used_symbols", []))),
        "data_snapshot_id": str(rerun.get("data_snapshot_id", data_fix.get("data_snapshot_id", ""))),
        "data_snapshot_hash": str(rerun.get("data_snapshot_hash", data_fix.get("data_snapshot_hash", ""))),
        "death_execution_rate": _extract_death_execution_rate(trace),
        "top_reject_reasons": top_reasons,
        "feasibility_min_required_risk_floor": risk_floor_by_tf,
        "global_baseline_metrics": dict(stage26.get("global_baseline_metrics", {})),
        "conditional_policy_metrics_live": dict(stage26.get("conditional_policy_metrics_live", {})),
        "contextual_edge_rows": list(context_verdict.get("rows", [])),
        "contextual_policy_verdict": str(context_verdict.get("policy_verdict", "")),
        "stage24_verdict": str(stage24_row.get("verdict", "")),
        "stage25_research_verdict": str(stage25_research_row.get("verdict", "")),
        "stage25_live_verdict": str(stage25_live_row.get("verdict", "")),
        "stage26_verdict": str(stage26_row.get("verdict", stage26.get("verdict", ""))),
        "final_verdict": final_verdict,
        "next_bottleneck": next_bottleneck,
        "next_actions": [
            "If ETH coverage drops again, rerun canonical downloader and keep BTC-only fallback explicit.",
            "If SIZE_TOO_SMALL or POLICY_CAP_HIT dominates, apply feasibility-driven risk floor per timeframe/equity tier.",
            "If only contextual edge exists, deploy context-gated policy rather than global always-on policy.",
        ],
    }
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(_render_md(payload), encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    report_md = Path(args.report_md)
    report_json = Path(args.report_json)
    payload = build_stage27_master_report(
        docs_dir=Path(args.docs_dir),
        report_md=report_md,
        report_json=report_json,
    )
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")
    print(f"final_verdict: {payload.get('final_verdict', '')}")


if __name__ == "__main__":
    main()
