"""Stage-27.9 full rerun orchestrator with contextual evidence aggregation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage27.context_evidence import evaluate_context_evidence, render_context_evidence_md
from buffmini.stage27.coverage_gate import evaluate_coverage_gate
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-27.9 full rerun suite")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-insufficient-data", action="store_true")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--timeframes", type=str, default="")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _csv(value: str, default: list[str]) -> list[str]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or list(default)


def _run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    done = subprocess.run(cmd, check=False)
    return int(done.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return {}


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-27.9 Rerun Report",
        "",
        f"- generated_at: `{payload.get('generated_at', '')}`",
        f"- coverage_gate_status: `{payload.get('coverage_gate_status', '')}`",
        f"- used_symbols: `{payload.get('used_symbols', [])}`",
        f"- data_snapshot_id: `{payload.get('data_snapshot_id', '')}`",
        "",
        "## Core Metrics",
        f"- zero_trade_pct: `{float(payload.get('zero_trade_pct', 0.0)):.6f}`",
        f"- invalid_pct: `{float(payload.get('invalid_pct', 0.0)):.6f}`",
        f"- death_execution_rate: `{float(payload.get('death_execution_rate', 0.0)):.6f}`",
        f"- walkforward_executed_true_pct: `{float(payload.get('walkforward_executed_true_pct', 0.0)):.6f}`",
        f"- mc_trigger_rate: `{float(payload.get('mc_trigger_rate', 0.0)):.6f}`",
        "",
        "## Top Reject Reasons",
    ]
    for item in payload.get("top_reject_reasons", [])[:10]:
        lines.append(f"- {item.get('reason','')}: {int(item.get('count', 0))}")

    lines.extend(
        [
            "",
            "## Rolling Context Edges",
            f"- robust: `{int((payload.get('rolling_context_edges', {}) or {}).get('counts', {}).get('ROBUST_CONTEXT_EDGE', 0))}`",
            f"- weak: `{int((payload.get('rolling_context_edges', {}) or {}).get('counts', {}).get('WEAK_CONTEXT_EDGE', 0))}`",
            f"- noise: `{int((payload.get('rolling_context_edges', {}) or {}).get('counts', {}).get('NOISE', 0))}`",
            "",
            "| stage | status | run_id |",
            "| --- | --- | --- |",
        ]
    )
    for row in payload.get("stages", []):
        lines.append(f"| {row.get('stage','')} | {row.get('status','')} | {row.get('run_id','')} |")

    lines.extend(["", "## Warnings"])
    warnings = list(payload.get("warnings", []))
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage26_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    symbols = _csv(args.symbols, list(stage26_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])))
    timeframes = _csv(args.timeframes, list(stage26_cfg.get("timeframes", ["15m", "30m", "1h", "2h", "4h"])))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    gate = evaluate_coverage_gate(
        config=cfg,
        symbols=symbols,
        timeframe=str(stage26_cfg.get("base_timeframe", "1m")),
        data_dir=args.data_dir,
        allow_insufficient_data=bool(args.allow_insufficient_data),
        auto_btc_fallback=True,
    )

    payload: dict[str, Any] = {
        "stage": "27.9.6",
        "generated_at": utc_now_compact(),
        "seed": int(args.seed),
        "dry_run": bool(args.dry_run),
        "coverage_gate_status": str(gate.status),
        "requested_symbols": list(gate.requested_symbols),
        "used_symbols": list(gate.used_symbols),
        "disabled_symbols": list(gate.disabled_symbols),
        "coverage_years_by_symbol": dict(gate.coverage_years_by_symbol),
        "warnings": [],
        **snapshot_metadata_from_config(cfg),
    }

    if not gate.can_run:
        payload.update(
            {
                "zero_trade_pct": 100.0,
                "invalid_pct": 100.0,
                "death_execution_rate": 0.0,
                "walkforward_executed_true_pct": 0.0,
                "mc_trigger_rate": 0.0,
                "top_reject_reasons": [],
                "rolling_context_edges": {"counts": {"ROBUST_CONTEXT_EDGE": 0, "WEAK_CONTEXT_EDGE": 0, "NOISE": 0}, "rows": []},
                "stages": [],
                "status": "INSUFFICIENT_DATA",
                "runtime_seconds": float(time.perf_counter() - started),
            }
        )
        (docs_dir / "stage27_9_rerun_report.md").write_text(_render_report(payload), encoding="utf-8")
        (docs_dir / "stage27_9_rerun_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"report_md: {docs_dir / 'stage27_9_rerun_report.md'}")
        print(f"report_json: {docs_dir / 'stage27_9_rerun_summary.json'}")
        raise SystemExit(2)

    py = sys.executable
    used_symbols_csv = ",".join(gate.used_symbols)
    timeframes_csv = ",".join(timeframes)
    dry_flag = ["--dry-run"] if bool(args.dry_run) else []
    allow_flag = ["--allow-insufficient-data"] if bool(args.allow_insufficient_data) else []
    warnings: list[str] = []

    stage_rows: list[dict[str, Any]] = []

    code24 = _run(
        [
            py,
            "scripts/run_stage24_audit.py",
            "--seed",
            str(int(args.seed)),
            "--symbols",
            used_symbols_csv,
            "--base-timeframe",
            str(stage26_cfg.get("base_timeframe", "1m")),
            "--operational-timeframe",
            "1h",
            "--docs-dir",
            str(docs_dir),
            *dry_flag,
            *allow_flag,
        ]
    )
    if code24 != 0:
        warnings.append(f"stage24_exit_code:{code24}")
    stage24 = _read_json(docs_dir / "stage24_report_summary.json")
    stage_rows.append({"stage": "24", "status": str(stage24.get("verdict", "")), "run_id": str((stage24.get("risk_pct_mode", {}) or {}).get("run_id", ""))})

    code25_research = _run(
        [
            py,
            "scripts/run_stage25B_edge_program.py",
            "--seed",
            str(int(args.seed)),
            "--mode",
            "research",
            "--symbols",
            used_symbols_csv,
            "--timeframes",
            timeframes_csv,
            "--docs-dir",
            str(docs_dir),
            *dry_flag,
            *allow_flag,
        ]
    )
    if code25_research != 0:
        warnings.append(f"stage25_research_exit_code:{code25_research}")
    stage25_research = _read_json(docs_dir / "stage25B_edge_program_summary.json")
    stage_rows.append({"stage": "25_research", "status": str(stage25_research.get("status", "")), "run_id": str(stage25_research.get("run_id", ""))})

    code25_live = _run(
        [
            py,
            "scripts/run_stage25B_edge_program.py",
            "--seed",
            str(int(args.seed)),
            "--mode",
            "live",
            "--symbols",
            used_symbols_csv,
            "--timeframes",
            timeframes_csv,
            "--docs-dir",
            str(docs_dir),
            *dry_flag,
            *allow_flag,
        ]
    )
    if code25_live != 0:
        warnings.append(f"stage25_live_exit_code:{code25_live}")
    stage25_live = _read_json(docs_dir / "stage25B_edge_program_summary.json")
    stage_rows.append({"stage": "25_live", "status": str(stage25_live.get("status", "")), "run_id": str(stage25_live.get("run_id", ""))})

    code26 = _run(
        [
            py,
            "scripts/run_stage26.py",
            "--seed",
            str(int(args.seed)),
            "--symbols",
            used_symbols_csv,
            "--timeframes",
            timeframes_csv,
            "--docs-dir",
            str(docs_dir),
            *dry_flag,
            *allow_flag,
        ]
    )
    if code26 != 0:
        warnings.append(f"stage26_exit_code:{code26}")
    stage26 = _read_json(docs_dir / "stage26_report_summary.json")
    stage_rows.append({"stage": "26", "status": str(stage26.get("verdict", "")), "run_id": str(stage26.get("run_id", ""))})

    code_roll = _run(
        [
            py,
            "scripts/run_stage27_9_rolling_discovery.py",
            "--seed",
            str(int(args.seed)),
            "--symbols",
            used_symbols_csv,
            "--timeframes",
            timeframes_csv,
            "--windows",
            "3m,6m",
            "--step-size",
            "1m",
            "--docs-dir",
            str(docs_dir),
            *dry_flag,
            *allow_flag,
        ]
    )
    if code_roll != 0:
        warnings.append(f"stage27_9_rolling_exit_code:{code_roll}")
    stage27_roll = _read_json(docs_dir / "stage27_9_rolling_summary.json")
    stage_rows.append({"stage": "27_9_rolling", "status": "OK" if code_roll == 0 else "FAILED", "run_id": str(stage27_roll.get("run_id", ""))})

    code_feas = _run(
        [
            py,
            "scripts/run_stage27_feasibility_audit.py",
            "--seed",
            str(int(args.seed)),
            "--symbols",
            used_symbols_csv,
            "--timeframes",
            timeframes_csv,
            "--docs-dir",
            str(docs_dir),
            *dry_flag,
            *allow_flag,
        ]
    )
    if code_feas != 0:
        warnings.append(f"stage27_feasibility_exit_code:{code_feas}")
    stage27_feas = _read_json(docs_dir / "stage27_feasibility_summary.json")
    stage_rows.append({"stage": "27_feasibility", "status": "OK" if code_feas == 0 else "FAILED", "run_id": str(stage27_feas.get("run_id", ""))})

    rolling_rows = pd.DataFrame()
    rolling_run_id = str(stage27_roll.get("run_id", ""))
    if rolling_run_id:
        rolling_path = Path("runs") / rolling_run_id / "stage27_9" / "rolling_results.csv"
        if rolling_path.exists():
            rolling_rows = pd.read_csv(rolling_path)
    rolling_context_edges = evaluate_context_evidence(rolling_rows)
    (docs_dir / "stage27_9_context_evidence.json").write_text(
        json.dumps(rolling_context_edges, indent=2, allow_nan=False), encoding="utf-8"
    )
    (docs_dir / "stage27_9_context_evidence.md").write_text(
        render_context_evidence_md(rolling_context_edges), encoding="utf-8"
    )

    stage26_live_metrics = dict(stage26.get("conditional_policy_metrics_live", {}))
    stage24_risk = dict(stage24.get("risk_pct_mode", {}))
    stage25_live_metrics = dict(stage25_live.get("metrics", {}))

    zero_trade_pct = float(
        stage26_live_metrics.get(
            "zero_trade_pct",
            stage25_live_metrics.get("zero_trade_pct", stage24_risk.get("zero_trade_pct", 0.0)),
        )
    )
    invalid_pct = float(
        stage26_live_metrics.get("invalid_pct", stage25_live_metrics.get("invalid_pct", stage24_risk.get("invalid_pct", 0.0)))
    )
    wf_pct = float(
        stage26_live_metrics.get(
            "walkforward_executed_true_pct",
            stage25_live_metrics.get("walkforward_executed_true_pct", stage24_risk.get("walkforward_executed_true_pct", 0.0)),
        )
    )
    mc_rate = float(
        stage26_live_metrics.get(
            "mc_trigger_rate",
            stage25_live_metrics.get("mc_trigger_rate", stage24_risk.get("mc_trigger_rate", 0.0)),
        )
    )

    next_bottleneck = dict(stage24.get("next_bottleneck", {}))
    death_execution_rate = float(next_bottleneck.get("death_rate", 0.0)) if str(next_bottleneck.get("gate", "")) == "death_execution" else 0.0
    top_reject_reasons = list(stage27_feas.get("top_reject_reasons", []))

    payload.update(
        {
            "stages": stage_rows,
            "zero_trade_pct": float(zero_trade_pct),
            "invalid_pct": float(invalid_pct),
            "walkforward_executed_true_pct": float(wf_pct),
            "mc_trigger_rate": float(mc_rate),
            "death_execution_rate": float(death_execution_rate),
            "top_reject_reasons": top_reject_reasons,
            "rolling_window_counts": dict(stage27_roll.get("window_counts", {})),
            "rolling_context_edges": rolling_context_edges,
            "runtime_seconds": float(time.perf_counter() - started),
            "warnings": warnings,
            "status": "OK" if not warnings else "WARNINGS",
        }
    )
    payload["summary_hash"] = stable_hash(
        {
            "stage_rows": stage_rows,
            "zero_trade_pct": payload["zero_trade_pct"],
            "invalid_pct": payload["invalid_pct"],
            "death_execution_rate": payload["death_execution_rate"],
            "rolling_counts": payload["rolling_window_counts"],
            "top_reject_reasons": top_reject_reasons,
        },
        length=16,
    )

    report_md = docs_dir / "stage27_9_rerun_report.md"
    report_json = docs_dir / "stage27_9_rerun_summary.json"
    report_md.write_text(_render_report(payload), encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")


if __name__ == "__main__":
    main()
