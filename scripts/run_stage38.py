"""Stage-38 orchestrator: trace, logic audit, and master conclusion."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage38.reporting import validate_stage38_master_summary
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-38 end-to-end logic audit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--budget-small", action="store_true")
    parser.add_argument("--stage28-run-id", type=str, default="")
    return parser.parse_args()


def _run_checked(cmd: list[str], *, label: str) -> tuple[str, str]:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    stdout = str(proc.stdout or "")
    stderr = str(proc.stderr or "")
    if int(proc.returncode) != 0:
        tail = "\n".join((stdout + "\n" + stderr).splitlines()[-80:])
        raise RuntimeError(f"{label} failed (exit={proc.returncode})\n{tail}")
    return stdout, stderr


def _parse_value(stdout: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(\S+)\s*$", stdout, flags=re.MULTILINE)
    return str(match.group(1)) if match else ""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_stage28_run_id(*, preferred: str, runs_dir: Path, docs_dir: Path) -> str:
    value = str(preferred).strip()
    if value:
        return value
    activation = _load_json(docs_dir / "stage37_activation_hunt_summary.json")
    candidate = str(activation.get("stage28_run_id", "")).strip()
    if candidate and (runs_dir / candidate / "stage28").exists():
        return candidate
    return ""


def _verdict(*, contradiction_fixed: bool, self_learning_rows: int, engine_raw_signal_count: int) -> str:
    if not contradiction_fixed:
        return "LOGIC_STILL_BROKEN"
    if self_learning_rows > 0 and engine_raw_signal_count <= 0:
        return "SELF_LEARNING_NOW_REAL_BUT_SIGNAL_WEAK"
    if engine_raw_signal_count <= 0:
        return "REPORTING_FIXED_ENGINE_ZERO_TRADE"
    return "LOGIC_FIXED_PARTIAL_PROGRESS"


def _render_master(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-38 Master Report",
        "",
        "## What Was Broken Before Stage-38",
        "- Activation-hunt counts were inflated by NaN active-candidate parsing and diverged from engine replay zero-signal reality.",
        "- OI short-only behavior existed in parts of the pipeline but runtime/report consistency was not explicit.",
        "- Self-learning memory could remain empty on zero-trade runs, preventing failure-aware evolution.",
        "",
        "## What Was Fixed",
        "- Added deterministic runtime trace from entrypoint to report artifacts.",
        "- Fixed hunt raw-signal parsing and added explicit composer lineage counts.",
        "- Added strict OI short-horizon runtime usage metadata and consistency checks.",
        "- Ensured self-learning registry persists failure motifs and elite flags even in zero-trade runs.",
        "",
        "## Logic Consistency",
        f"- stage28_run_id: `{payload.get('stage28_run_id', '')}`",
        f"- trace_hash: `{payload.get('trace_hash', '')}`",
        f"- contradiction_fixed: `{bool(payload.get('contradiction_fixed', False))}`",
        f"- collapse_reason: `{payload.get('collapse_reason', '')}`",
        "",
        "## OI Short-Only",
        f"- short_only_enabled: `{bool(payload.get('oi_usage', {}).get('short_only_enabled', False))}`",
        f"- timeframe: `{payload.get('oi_usage', {}).get('timeframe', '')}`",
        f"- timeframe_allowed: `{bool(payload.get('oi_usage', {}).get('timeframe_allowed', False))}`",
        f"- oi_active_runtime: `{bool(payload.get('oi_usage', {}).get('oi_active_runtime', False))}`",
        "",
        "## Self-Learning Memory",
        f"- registry_rows: `{int(payload.get('self_learning', {}).get('registry_rows', 0))}`",
        f"- elites_count: `{int(payload.get('self_learning', {}).get('elites_count', 0))}`",
        f"- dead_family_count: `{int(payload.get('self_learning', {}).get('dead_family_count', 0))}`",
        f"- failure_motif_tags_non_empty: `{bool(payload.get('self_learning', {}).get('failure_motif_tags_non_empty', False))}`",
        "",
        "## Engine State",
        f"- engine_raw_signal_count: `{int(payload.get('lineage_table', {}).get('engine_raw_signal_count', 0))}`",
        f"- final_trade_count: `{float(payload.get('lineage_table', {}).get('final_trade_count', 0.0)):.6f}`",
        f"- still_no_edge: `{bool(int(payload.get('lineage_table', {}).get('engine_raw_signal_count', 0)) <= 0)}`",
        "",
        "## Remaining Bottleneck",
        f"- biggest_remaining_bottleneck: `{payload.get('biggest_remaining_bottleneck', '')}`",
        f"- next_cheapest_high_confidence_action: `{payload.get('next_action', '')}`",
        "",
        "## Verdict",
        f"- `{payload.get('verdict', '')}`",
        f"- summary_hash: `{payload.get('summary_hash', '')}`",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)

    stage28_run_id = _resolve_stage28_run_id(
        preferred=str(args.stage28_run_id),
        runs_dir=runs_dir,
        docs_dir=docs_dir,
    )

    trace_cmd = [
        sys.executable,
        "scripts/stage38_trace_run.py",
        "--config",
        str(args.config),
        "--seed",
        str(int(args.seed)),
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
        "--mode",
        "both",
    ]
    if stage28_run_id:
        trace_cmd.extend(["--stage28-run-id", stage28_run_id])
    if bool(args.budget_small):
        trace_cmd.append("--budget-small")
    trace_stdout, _ = _run_checked(trace_cmd, label="stage38_trace_run")
    stage28_run_id = _parse_value(trace_stdout, "stage28_run_id") or stage28_run_id
    if not stage28_run_id:
        raise RuntimeError("unable to resolve stage28_run_id")

    activation_cmd = [
        sys.executable,
        "scripts/stage37_activation_hunt.py",
        "--config",
        str(args.config),
        "--seed",
        str(int(args.seed)),
        "--mode",
        "both",
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
        "--stage28-run-id",
        stage28_run_id,
    ]
    if bool(args.budget_small):
        activation_cmd.append("--budget-small")
    _run_checked(activation_cmd, label="stage37_activation_hunt")

    self_learning_cmd = [
        sys.executable,
        "scripts/stage37_self_learning_rerun.py",
        "--config",
        str(args.config),
        "--seed",
        str(int(args.seed)),
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
        "--stage28-run-id",
        stage28_run_id,
        "--activation-summary",
        str(docs_dir / "stage37_activation_hunt_summary.json"),
    ]
    _run_checked(self_learning_cmd, label="stage37_self_learning_rerun")

    logic_cmd = [
        sys.executable,
        "scripts/stage38_logic_audit.py",
        "--stage28-run-id",
        stage28_run_id,
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
        "--config",
        str(args.config),
    ]
    _run_checked(logic_cmd, label="stage38_logic_audit")

    trace_payload = _load_json(runs_dir / stage28_run_id / "stage38" / "stage38_trace.json")
    logic_payload = _load_json(docs_dir / "stage38_logic_audit_summary.json")
    stage28_summary = _load_json(runs_dir / stage28_run_id / "stage28" / "summary.json")
    cfg = load_config(Path(args.config))

    line = dict(logic_payload.get("lineage_table", {}))
    oi_usage = dict(logic_payload.get("oi_usage", {}))
    self_learning = dict(logic_payload.get("self_learning", {}))
    contradiction_fixed = bool(logic_payload.get("contradiction_fixed", False))
    engine_raw_signal_count = int(line.get("engine_raw_signal_count", 0))
    verdict = _verdict(
        contradiction_fixed=contradiction_fixed,
        self_learning_rows=int(self_learning.get("registry_rows", 0)),
        engine_raw_signal_count=engine_raw_signal_count,
    )
    bottleneck = str(stage28_summary.get("next_bottleneck", "")) or str(logic_payload.get("collapse_reason", ""))
    next_action = (
        "Increase upstream candidate signal quality (family/context generation), then rerun Stage-37/38 lineage checks."
        if engine_raw_signal_count <= 0
        else "Tune policy thresholds and cost gates with lineage table as guardrails."
    )

    master_payload = {
        "stage": "38.6",
        "seed": int(args.seed),
        "config_path": str(args.config),
        "config_hash": stable_hash(cfg, length=16),
        "stage28_run_id": stage28_run_id,
        "trace_hash": str(trace_payload.get("trace_hash", "")),
        "lineage_table": line,
        "collapse_reason": str(logic_payload.get("collapse_reason", "")),
        "contradiction_fixed": bool(contradiction_fixed),
        "oi_usage": oi_usage,
        "self_learning": self_learning,
        "verdict": verdict,
        "biggest_remaining_bottleneck": bottleneck,
        "next_action": next_action,
        "report_paths": {
            "flow_report": str((docs_dir / "stage38_end_to_end_flow_report.md").as_posix()),
            "logic_report": str((docs_dir / "stage38_logic_audit_report.md").as_posix()),
            "test_hardening_report": str((docs_dir / "stage38_test_hardening_report.md").as_posix()),
        },
    }
    master_payload["summary_hash"] = stable_hash(
        {
            "stage28_run_id": master_payload["stage28_run_id"],
            "trace_hash": master_payload["trace_hash"],
            "lineage_table": master_payload["lineage_table"],
            "collapse_reason": master_payload["collapse_reason"],
            "contradiction_fixed": master_payload["contradiction_fixed"],
            "oi_usage": master_payload["oi_usage"],
            "self_learning": master_payload["self_learning"],
            "verdict": master_payload["verdict"],
            "bottleneck": master_payload["biggest_remaining_bottleneck"],
        },
        length=16,
    )
    validate_stage38_master_summary(master_payload)

    (docs_dir / "stage38_master_summary.json").write_text(
        json.dumps(master_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (docs_dir / "stage38_master_report.md").write_text(_render_master(master_payload), encoding="utf-8")

    print(f"stage28_run_id: {stage28_run_id}")
    print(f"summary_hash: {master_payload['summary_hash']}")
    print(f"verdict: {master_payload['verdict']}")
    print(f"master_report: {docs_dir / 'stage38_master_report.md'}")
    print(f"master_summary: {docs_dir / 'stage38_master_summary.json'}")


if __name__ == "__main__":
    main()

