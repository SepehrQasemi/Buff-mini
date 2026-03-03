"""Stage-27.5 unified rerun orchestrator for Stage-24/25/26 + rolling discovery."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage27.coverage_gate import evaluate_coverage_gate
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-27.5 unified rerun suite")
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


def _run_command(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-27 Rerun Report",
        "",
        f"- generated_at: `{payload.get('generated_at', '')}`",
        f"- coverage_gate_status: `{payload.get('coverage_gate_status', '')}`",
        f"- used_symbols: `{payload.get('used_symbols', [])}`",
        f"- data_snapshot_id: `{payload.get('data_snapshot_id', '')}`",
        "",
        "| stage | run_id | verdict | exp_lcb | trade_count | zero_trade_pct |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload.get("stages", []):
        lines.append(
            f"| {row.get('stage','')} | {row.get('run_id','')} | {row.get('verdict','')} | "
            f"{float(row.get('exp_lcb',0.0)):.6f} | {float(row.get('trade_count',0.0)):.2f} | {float(row.get('zero_trade_pct',0.0)):.2f} |"
        )
    lines.extend(["", "## Warnings"])
    for item in payload.get("warnings", []):
        lines.append(f"- {item}")
    return "\n".join(lines).strip() + "\n"


def run_stage27_rerun_all(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    stage26_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    symbols = _csv(args.symbols, list(stage26_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])))
    timeframes = _csv(args.timeframes, list(stage26_cfg.get("timeframes", ["15m", "30m", "1h", "2h", "4h"])))
    gate = evaluate_coverage_gate(
        config=cfg,
        symbols=symbols,
        timeframe=str(stage26_cfg.get("base_timeframe", "1m")),
        data_dir=args.data_dir,
        allow_insufficient_data=bool(args.allow_insufficient_data),
        auto_btc_fallback=True,
    )
    started = time.perf_counter()
    warnings: list[str] = []
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    payload: dict[str, Any] = {
        "stage": "27.5",
        "generated_at": utc_now_compact(),
        "seed": int(args.seed),
        "dry_run": bool(args.dry_run),
        "coverage_gate_status": str(gate.status),
        "requested_symbols": list(gate.requested_symbols),
        "used_symbols": list(gate.used_symbols),
        "disabled_symbols": list(gate.disabled_symbols),
        "coverage_years_by_symbol": dict(gate.coverage_years_by_symbol),
        "warnings": warnings,
        **snapshot_metadata_from_config(cfg),
    }
    if not gate.can_run:
        payload["verdict"] = "INSUFFICIENT_DATA"
        payload["runtime_seconds"] = float(time.perf_counter() - started)
        report_md = docs_dir / "stage27_rerun_report.md"
        report_json = docs_dir / "stage27_rerun_summary.json"
        report_md.write_text(_render_report({**payload, "stages": []}), encoding="utf-8")
        report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        return payload

    used_symbols_csv = ",".join(gate.used_symbols)
    timeframes_csv = ",".join(timeframes)
    dry_flag = ["--dry-run"] if bool(args.dry_run) else ["--use-real-data"]
    allow_flag = ["--allow-insufficient-data"] if bool(args.allow_insufficient_data) else []

    code24 = _run_command(
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

    code25r = _run_command(
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
            *(["--dry-run"] if bool(args.dry_run) else []),
            *allow_flag,
        ]
    )
    if code25r != 0:
        warnings.append(f"stage25_research_exit_code:{code25r}")
    stage25_research = _read_json(docs_dir / "stage25B_edge_program_summary.json")

    code25l = _run_command(
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
            *(["--dry-run"] if bool(args.dry_run) else []),
            *allow_flag,
        ]
    )
    if code25l != 0:
        warnings.append(f"stage25_live_exit_code:{code25l}")
    stage25_live = _read_json(docs_dir / "stage25B_edge_program_summary.json")

    code26 = _run_command(
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
            *(["--dry-run"] if bool(args.dry_run) else []),
            *allow_flag,
        ]
    )
    if code26 != 0:
        warnings.append(f"stage26_exit_code:{code26}")
    stage26 = _read_json(docs_dir / "stage26_report_summary.json")

    code27roll = _run_command(
        [
            py,
            "scripts/run_stage27_rolling_discovery.py",
            "--seed",
            str(int(args.seed)),
            "--symbols",
            used_symbols_csv,
            "--timeframes",
            timeframes_csv,
            "--docs-dir",
            str(docs_dir),
            *(["--dry-run"] if bool(args.dry_run) else []),
            *allow_flag,
        ]
    )
    if code27roll != 0:
        warnings.append(f"stage27_rolling_exit_code:{code27roll}")
    stage27_roll = _read_json(docs_dir / "stage27_research_engine_summary.json")

    stages = [
        {
            "stage": "24",
            "run_id": str((stage24.get("risk_pct_mode", {}) or {}).get("run_id", "")),
            "verdict": str(stage24.get("verdict", "")),
            "exp_lcb": float((stage24.get("risk_pct_mode", {}) or {}).get("exp_lcb", 0.0)),
            "trade_count": float((stage24.get("risk_pct_mode", {}) or {}).get("trade_count", 0.0)),
            "zero_trade_pct": float((stage24.get("risk_pct_mode", {}) or {}).get("zero_trade_pct", 0.0)),
        },
        {
            "stage": "25_research",
            "run_id": str(stage25_research.get("run_id", "")),
            "verdict": str(stage25_research.get("status", "")),
            "exp_lcb": float((stage25_research.get("metrics", {}) or {}).get("exp_lcb_best", 0.0)),
            "trade_count": float((stage25_research.get("metrics", {}) or {}).get("trade_count_total", 0.0)),
            "zero_trade_pct": float((stage25_research.get("metrics", {}) or {}).get("zero_trade_pct", 0.0)),
        },
        {
            "stage": "25_live",
            "run_id": str(stage25_live.get("run_id", "")),
            "verdict": str(stage25_live.get("status", "")),
            "exp_lcb": float((stage25_live.get("metrics", {}) or {}).get("exp_lcb_best", 0.0)),
            "trade_count": float((stage25_live.get("metrics", {}) or {}).get("trade_count_total", 0.0)),
            "zero_trade_pct": float((stage25_live.get("metrics", {}) or {}).get("zero_trade_pct", 0.0)),
        },
        {
            "stage": "26",
            "run_id": str(stage26.get("run_id", "")),
            "verdict": str(stage26.get("verdict", "")),
            "exp_lcb": float((stage26.get("conditional_policy_metrics_live", {}) or {}).get("exp_lcb", 0.0)),
            "trade_count": float((stage26.get("conditional_policy_metrics_live", {}) or {}).get("trade_count", 0.0)),
            "zero_trade_pct": float((stage26.get("conditional_policy_metrics_live", {}) or {}).get("zero_trade_pct", 0.0)),
        },
        {
            "stage": "27_rolling",
            "run_id": str(stage27_roll.get("run_id", "")),
            "verdict": "ROLLING_DISCOVERY",
            "exp_lcb": 0.0,
            "trade_count": float(stage27_roll.get("rows", 0)),
            "zero_trade_pct": 0.0,
        },
    ]
    payload["stages"] = stages
    payload["runtime_seconds"] = float(time.perf_counter() - started)

    report_md = docs_dir / "stage27_rerun_report.md"
    report_json = docs_dir / "stage27_rerun_summary.json"
    report_md.write_text(_render_report(payload), encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    payload = run_stage27_rerun_all(args)
    print(f"report_md: {Path(args.docs_dir) / 'stage27_rerun_report.md'}")
    print(f"report_json: {Path(args.docs_dir) / 'stage27_rerun_summary.json'}")
    print(f"coverage_gate_status: {payload.get('coverage_gate_status', '')}")


if __name__ == "__main__":
    main()
