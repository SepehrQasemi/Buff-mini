"""Stage-26.9.5 unified rerun orchestrator for Stage-24/25/26/15.9 on local data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage25.edge_program import run_stage25b_edge_program


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-26.9.5 full rerun suite")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _run_subprocess(command: list[str]) -> int:
    print("$", " ".join(command))
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def _build_stage_payloads(
    *,
    stage24: dict[str, Any],
    stage25_research: dict[str, Any],
    stage25_live: dict[str, Any],
    stage26: dict[str, Any],
    trace: dict[str, Any],
    coverage: dict[str, Any],
    previous: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    stage24_metrics = dict(stage24.get("risk_pct_mode", {}))
    stage25_research_metrics = dict(stage25_research.get("metrics", {}))
    stage25_live_metrics = dict(stage25_live.get("metrics", {}))

    payload = {
        "stage": "26.9.5",
        "data_coverage_years_used": dict(coverage.get("coverage_years_per_symbol", {})),
        "coverage_meets_4y": all(float(v) >= 4.0 for v in dict(coverage.get("coverage_years_per_symbol", {})).values()) if coverage else False,
        "stages": {
            "stage24": {
                "run_id": str(stage24_metrics.get("run_id", "")),
                "timeframes_used": ["1h"],
                "verdict": str(stage24.get("verdict", "")),
                "metrics": {
                    "exp_lcb": float(stage24_metrics.get("exp_lcb", 0.0)),
                    "PF": float(stage24_metrics.get("PF_clipped", 0.0)),
                    "maxDD": float(stage24_metrics.get("maxDD", 0.0)),
                    "trade_count": float(stage24_metrics.get("trade_count", 0.0)),
                    "zero_trade_pct": float(stage24_metrics.get("zero_trade_pct", 0.0)),
                },
                "reject_breakdown_top3": list((stage24_metrics.get("top_reject_reasons") or [])[:3]),
            },
            "stage25_research": {
                "run_id": str(stage25_research.get("run_id", "")),
                "timeframes_used": list(stage25_research.get("timeframes", [])),
                "verdict": str(stage25_research.get("status", "")),
                "metrics": {
                    "exp_lcb_best": float(stage25_research_metrics.get("exp_lcb_best", 0.0)),
                    "exp_lcb_median": float(stage25_research_metrics.get("exp_lcb_median", 0.0)),
                    "trade_count_total": float(stage25_research_metrics.get("trade_count_total", 0.0)),
                    "zero_trade_pct": float(stage25_research_metrics.get("zero_trade_pct", 0.0)),
                },
                "reject_breakdown_top3": [],
            },
            "stage25_live": {
                "run_id": str(stage25_live.get("run_id", "")),
                "timeframes_used": list(stage25_live.get("timeframes", [])),
                "verdict": str(stage25_live.get("status", "")),
                "metrics": {
                    "exp_lcb_best": float(stage25_live_metrics.get("exp_lcb_best", 0.0)),
                    "exp_lcb_median": float(stage25_live_metrics.get("exp_lcb_median", 0.0)),
                    "trade_count_total": float(stage25_live_metrics.get("trade_count_total", 0.0)),
                    "zero_trade_pct": float(stage25_live_metrics.get("zero_trade_pct", 0.0)),
                },
                "reject_breakdown_top3": [],
            },
            "stage26": {
                "run_id": str(stage26.get("run_id", "")),
                "timeframes_used": list(stage26.get("timeframes_tested", [])),
                "verdict": str(stage26.get("verdict", "")),
                "metrics": {
                    "exp_lcb": float(stage26.get("conditional_policy_metrics_live", {}).get("exp_lcb", 0.0)),
                    "PF": float(stage26.get("conditional_policy_metrics_live", {}).get("PF_clipped", 0.0)),
                    "maxDD": float(stage26.get("conditional_policy_metrics_live", {}).get("maxDD", 0.0)),
                    "trade_count": float(stage26.get("conditional_policy_metrics_live", {}).get("trade_count", 0.0)),
                    "zero_trade_pct": float(stage26.get("conditional_policy_metrics_live", {}).get("zero_trade_pct", 0.0)),
                },
                "reject_breakdown_top3": [
                    {"reason": k, "count": int(v)}
                    for k, v in list(dict(stage26.get("shadow_live_top_reasons", {})).items())[:3]
                ],
            },
            "stage15_9": {
                "run_id": str(trace.get("before_after", {}).get("post", {}).get("run_id", trace.get("run_id", ""))),
                "timeframes_used": [row.get("timeframe", "") for row in list(trace.get("per_tf", []))],
                "verdict": "TRACE_RECORDED",
                "metrics": {
                    "exp_lcb": 0.0,
                    "PF": 0.0,
                    "maxDD": 0.0,
                    "trade_count": 0.0,
                    "zero_trade_pct": float(trace.get("before_after", {}).get("post", {}).get("zero_trade_pct", 0.0)),
                },
                "reject_breakdown_top3": list(trace.get("top_bottlenecks", []))[:3],
            },
        },
    }

    changed: dict[str, bool] = {}
    prev24 = previous.get("stage24", {})
    prev25 = previous.get("stage25", {})
    prev26 = previous.get("stage26", {})
    changed["stage24_verdict_changed"] = str(prev24.get("verdict", "")) != str(stage24.get("verdict", ""))
    changed["stage25_verdict_changed"] = str(prev25.get("status", "")) != str(stage25_live.get("status", ""))
    changed["stage26_verdict_changed"] = str(prev26.get("verdict", "")) != str(stage26.get("verdict", ""))
    payload["changed_vs_previous"] = changed
    return payload


def _render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-26.9 Rerun Report",
        "",
        f"- coverage_meets_4y: `{bool(payload.get('coverage_meets_4y', False))}`",
        f"- data_coverage_years_used: `{payload.get('data_coverage_years_used', {})}`",
        "",
        "| stage | run_id | verdict | exp_lcb | PF | maxDD | trade_count | zero_trade_pct |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage_name, stage_payload in dict(payload.get("stages", {})).items():
        metrics = dict(stage_payload.get("metrics", {}))
        lines.append(
            "| "
            f"{stage_name} | {stage_payload.get('run_id','')} | {stage_payload.get('verdict','')} | "
            f"{float(metrics.get('exp_lcb', 0.0)):.6f} | {float(metrics.get('PF', 0.0)):.6f} | "
            f"{float(metrics.get('maxDD', 0.0)):.6f} | {float(metrics.get('trade_count', 0.0)):.2f} | "
            f"{float(metrics.get('zero_trade_pct', 0.0)):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Verdict Changes vs Previous",
            f"- stage24_verdict_changed: `{payload.get('changed_vs_previous', {}).get('stage24_verdict_changed', False)}`",
            f"- stage25_verdict_changed: `{payload.get('changed_vs_previous', {}).get('stage25_verdict_changed', False)}`",
            f"- stage26_verdict_changed: `{payload.get('changed_vs_previous', {}).get('stage26_verdict_changed', False)}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    previous = {
        "stage24": _read_json(docs_dir / "stage24_report_summary.json"),
        "stage25": _read_json(docs_dir / "stage25B_edge_program_summary.json"),
        "stage26": _read_json(docs_dir / "stage26_report_summary.json"),
        "stage15_9": _read_json(docs_dir / "stage15_9_signal_flow_bottleneck_summary.json"),
    }

    seed = int(args.seed)
    py = sys.executable

    code24 = _run_subprocess([py, "scripts/run_stage24_audit.py", "--seed", str(seed), "--use-real-data"])

    cfg = load_config(args.config)
    stage25_research = run_stage25b_edge_program(
        config=cfg,
        seed=seed,
        dry_run=False,
        mode="research",
        symbols=_csv(args.symbols),
        timeframes=_csv(args.timeframes),
        families=["price", "volatility", "flow"],
        composers=["weighted_sum"],
        cost_levels=["realistic", "high"],
        runs_root=args.runs_dir,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
        docs_dir=docs_dir,
        out_run_id=None,
    )
    stage25_live = run_stage25b_edge_program(
        config=cfg,
        seed=seed,
        dry_run=False,
        mode="live",
        symbols=_csv(args.symbols),
        timeframes=_csv(args.timeframes),
        families=["price", "volatility", "flow"],
        composers=["weighted_sum"],
        cost_levels=["realistic", "high"],
        runs_root=args.runs_dir,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
        docs_dir=docs_dir,
        out_run_id=None,
    )

    code26 = _run_subprocess([py, "scripts/run_stage26.py", "--seed", str(seed)])
    code159 = _run_subprocess([py, "scripts/trace_signal_flow.py", "--seed", str(seed)])

    stage24_summary = _read_json(docs_dir / "stage24_report_summary.json")
    stage26_summary = _read_json(docs_dir / "stage26_report_summary.json")
    trace_summary = _read_json(docs_dir / "stage15_9_signal_flow_bottleneck_summary.json")
    coverage = _read_json(docs_dir / "stage26_9_data_master_summary.json")

    payload = _build_stage_payloads(
        stage24=stage24_summary,
        stage25_research=dict(stage25_research.get("summary", {})),
        stage25_live=dict(stage25_live.get("summary", {})),
        stage26=stage26_summary,
        trace=trace_summary,
        coverage=coverage,
        previous=previous,
    )
    payload["runtime_seconds"] = float(time.perf_counter() - started)
    payload["exit_codes"] = {
        "stage24_cmd": int(code24),
        "stage26_cmd": int(code26),
        "stage15_9_cmd": int(code159),
    }

    md_path = docs_dir / "stage26_9_rerun_report.md"
    json_path = docs_dir / "stage26_9_rerun_summary.json"
    md_path.write_text(_render_md(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"report_md: {md_path}")
    print(f"report_json: {json_path}")


if __name__ == "__main__":
    main()
