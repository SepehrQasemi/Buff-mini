"""Stage-35.7 engine rerun orchestrator and multi-seed robustness gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR


REPORT_MD = Path("docs") / "stage35_7_engine_rerun_report.md"
REPORT_JSON = Path("docs") / "stage35_7_engine_rerun_summary.json"
MASTER_DOWNLOAD_SUMMARY = Path("docs") / "stage35_7_report_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-35.7 engine rerun and optional multi-seed robustness")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def _run(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": int(completed.returncode),
        "stdout": str(completed.stdout or ""),
        "stderr": str(completed.stderr or ""),
    }


def _snippet(text: str, limit: int = 1200) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]..."


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _is_promising(stage34_summary: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    research = float(stage34_summary.get("research_best_exp_lcb", 0.0) or 0.0)
    live = float(stage34_summary.get("live_best_exp_lcb", 0.0) or 0.0)
    trade_count = float(stage34_summary.get("trade_count", 0.0) or 0.0)
    promising = (max(research, live) > 0.0) and (trade_count > 10.0)
    return promising, {"research_best_exp_lcb": research, "live_best_exp_lcb": live, "trade_count": trade_count}


def _write_report(payload: dict[str, Any]) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Stage-35.7 Engine Rerun Report",
        "",
        f"- status: `{payload.get('status', '')}`",
        f"- reason: `{payload.get('reason', '')}`",
        f"- stage35_status: `{payload.get('stage35_status', '')}`",
        f"- promising: `{payload.get('promising', False)}`",
        f"- biggest_blocker: `{payload.get('biggest_blocker', '')}`",
        "",
        "## Metrics",
        f"- baseline: `{payload.get('baseline_metrics', {})}`",
        f"- after: `{payload.get('after_metrics', {})}`",
        f"- multi_seed_count: `{len(payload.get('multi_seed_runs', []))}`",
        "",
        "## Next Actions",
    ]
    actions = payload.get("next_actions", [])
    if isinstance(actions, list) and actions:
        for action in actions:
            lines.append(f"- `{action}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Command Evidence",
            "### stage35 stdout",
            "```text",
            _snippet(str(payload.get("stage35_stdout", ""))),
            "```",
            "### stage35 stderr",
            "```text",
            _snippet(str(payload.get("stage35_stderr", ""))),
            "```",
        ]
    )
    REPORT_MD.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_engine_rerun(args: argparse.Namespace) -> dict[str, Any]:
    download_summary = _load_json(MASTER_DOWNLOAD_SUMMARY)
    stage35_status = str(download_summary.get("status", "MISSING_STAGE35_7_REPORT"))
    coverage_ok = bool(download_summary.get("coverage_ok", False))
    payload: dict[str, Any] = {
        "status": "NOT_EXECUTED",
        "reason": "",
        "stage35_status": stage35_status,
        "promising": False,
        "biggest_blocker": "",
        "next_actions": [],
        "baseline_metrics": {},
        "after_metrics": {},
        "multi_seed_runs": [],
        "stage35_stdout": "",
        "stage35_stderr": "",
    }

    if not download_summary:
        payload["reason"] = "stage35_7_report_summary_missing"
        payload["biggest_blocker"] = "Stage-35.7 download summary missing"
        payload["next_actions"] = [
            "python scripts/run_stage35_real_download.py --config configs/local_coinapi.yaml --seed 42",
        ]
        _write_report(payload)
        return payload
    if not coverage_ok:
        payload["reason"] = "coverage_insufficient_or_download_blocked"
        payload["biggest_blocker"] = "Coverage gate not satisfied for engine rerun"
        payload["next_actions"] = [
            "python scripts/run_stage35_real_download.py --config configs/local_coinapi.yaml --seed 42",
        ]
        _write_report(payload)
        return payload

    cmd = [
        sys.executable,
        "scripts/run_stage35.py",
        "--config",
        str(Path(args.config)),
        "--seed",
        str(int(args.seed)),
        "--runs-dir",
        str(Path(args.runs_dir)),
    ]
    out = _run(cmd)
    payload["stage35_stdout"] = out["stdout"]
    payload["stage35_stderr"] = out["stderr"]
    if out["returncode"] != 0:
        payload["reason"] = "stage35_orchestrator_failed"
        payload["biggest_blocker"] = "Stage-35 orchestrator command failed"
        _write_report(payload)
        return payload

    stage34_summary = _load_json(Path("docs") / "stage34_report_summary.json")
    promising, metrics = _is_promising(stage34_summary)
    payload["after_metrics"] = metrics
    payload["baseline_metrics"] = {}
    payload["promising"] = promising

    if not promising:
        payload["status"] = "NO_EDGE"
        payload["reason"] = "promising_gate_not_met"
        payload["biggest_blocker"] = "No positive edge under promising gate"
        payload["next_actions"] = [
            "Review docs/stage35_report_summary.json and docs/stage34_report_summary.json for bottlenecks",
        ]
        _write_report(payload)
        return payload

    multi_seed_runs: list[dict[str, Any]] = []
    for seed in (43, 44, 45, 46, 47):
        run = _run(
            [
                sys.executable,
                "scripts/run_stage35.py",
                "--config",
                str(Path(args.config)),
                "--seed",
                str(int(seed)),
                "--runs-dir",
                str(Path(args.runs_dir)),
            ]
        )
        multi_seed_runs.append(
            {
                "seed": int(seed),
                "returncode": int(run["returncode"]),
                "stdout_tail": _snippet(str(run["stdout"]), limit=300),
                "stderr_tail": _snippet(str(run["stderr"]), limit=300),
            }
        )
    payload["multi_seed_runs"] = multi_seed_runs
    payload["status"] = "PROMISING_MULTI_SEED_DONE"
    payload["reason"] = "promising_gate_met"
    payload["biggest_blocker"] = ""
    payload["next_actions"] = []
    _write_report(payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_engine_rerun(args)
    print(f"status: {payload.get('status', '')}")
    print(f"report: {REPORT_MD.as_posix()}")
    print(f"summary: {REPORT_JSON.as_posix()}")


if __name__ == "__main__":
    main()

