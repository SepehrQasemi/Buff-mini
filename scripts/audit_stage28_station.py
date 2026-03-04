"""Stage-28 station safety audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage-28 station safety checks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return {}


def _check(name: str, passed: bool, details: str) -> dict[str, Any]:
    return {
        "check": str(name),
        "passed": bool(passed),
        "details": str(details),
    }


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=str(cwd) if cwd else None)


def _render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-28 Station Safety Audit",
        "",
        f"- generated_at: `{payload.get('generated_at', '')}`",
        f"- seed: `{payload.get('seed', 42)}`",
        f"- status: `{payload.get('status', '')}`",
        "",
        "| check | passed | details |",
        "| --- | --- | --- |",
    ]
    for row in payload.get("checks", []):
        lines.append(
            f"| {row.get('check', '')} | {bool(row.get('passed', False))} | {str(row.get('details', '')).replace('|', '/')} |"
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    checks: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="stage28_audit_") as tmp:
        root = Path(tmp)
        docs_research_a = root / "docs_research_a"
        docs_research_b = root / "docs_research_b"
        docs_live = root / "docs_live"
        runs_research_a = root / "runs_research_a"
        runs_research_b = root / "runs_research_b"
        runs_live = root / "runs_live"

        common = [
            "--seed",
            str(int(args.seed)),
            "--dry-run",
            "--budget-small",
            "--symbols",
            "BTC/USDT",
            "--timeframes",
            "1h",
            "--windows",
            "3m,6m",
            "--step-months",
            "1",
        ]

        run_a = _run([sys.executable, "scripts/run_stage28.py", "--mode", "research", *common, "--docs-dir", str(docs_research_a), "--runs-dir", str(runs_research_a)])
        checks.append(_check("stage28_research_smoke", run_a.returncode == 0, f"code={run_a.returncode}"))

        run_b = _run([sys.executable, "scripts/run_stage28.py", "--mode", "research", *common, "--docs-dir", str(docs_research_b), "--runs-dir", str(runs_research_b)])
        checks.append(_check("stage28_research_repeat", run_b.returncode == 0, f"code={run_b.returncode}"))

        run_live = _run([sys.executable, "scripts/run_stage28.py", "--mode", "live", *common, "--docs-dir", str(docs_live), "--runs-dir", str(runs_live)])
        checks.append(_check("stage28_live_smoke", run_live.returncode == 0, f"code={run_live.returncode}"))

        summary_a = _read_json(docs_research_a / "stage28_master_summary.json")
        summary_b = _read_json(docs_research_b / "stage28_master_summary.json")
        summary_live = _read_json(docs_live / "stage28_master_summary.json")

        hash_a = str(summary_a.get("summary_hash", ""))
        hash_b = str(summary_b.get("summary_hash", ""))
        checks.append(_check("determinism_summary_hash", bool(hash_a and hash_a == hash_b), f"hash_a={hash_a} hash_b={hash_b}"))

        cov_status = str(summary_a.get("coverage_gate_status", ""))
        checks.append(_check("coverage_gate_present", bool(cov_status), f"status={cov_status}"))

        split_ok = str(summary_a.get("mode", "")) == "research" and str(summary_live.get("mode", "")) == "live"
        checks.append(_check("research_live_split_mode", split_ok, f"research={summary_a.get('mode')} live={summary_live.get('mode')}"))

        live_constraints_ok = "shadow_live_reject_rate" in summary_a and "shadow_live_reject_rate" in summary_live
        checks.append(_check("shadow_live_reject_rate_present", live_constraints_ok, f"research={summary_a.get('shadow_live_reject_rate')} live={summary_live.get('shadow_live_reject_rate')}"))

        wc = dict(summary_a.get("window_counts", {}))
        wc_ok = True
        wc_details: list[str] = []
        for key, item in wc.items():
            generated = int((item or {}).get("generated", 0))
            evaluated = int((item or {}).get("evaluated", 0))
            expected = int((item or {}).get("expected", 0))
            if generated < 0 or evaluated < 0 or expected < 0 or evaluated > generated:
                wc_ok = False
            wc_details.append(f"{key}:g={generated},e={evaluated},x={expected}")
        checks.append(_check("window_counts_consistency", wc_ok, "; ".join(wc_details) if wc_details else "no_windows"))

        leak_tests = _run([sys.executable, "-m", "pytest", "-q", "tests/test_stage26_context_no_leakage.py", "tests/test_stage28_context_mask_correctness.py"])
        checks.append(_check("leakage_guards", leak_tests.returncode == 0, f"code={leak_tests.returncode}"))

        required_docs = [
            Path("docs/stage28_master_report.md"),
            Path("docs/stage28_master_summary.json"),
            Path("docs/stage28_product_spec.md"),
            Path("docs/stage28_test_plan.md"),
        ]
        missing = [str(path.as_posix()) for path in required_docs if not path.exists()]
        checks.append(_check("required_docs_exist", len(missing) == 0, "missing=" + ",".join(missing) if missing else "ok"))

    passed = int(sum(1 for row in checks if bool(row.get("passed", False))))
    status = "PASS" if passed == len(checks) else "FAIL"
    payload = {
        "stage": "28.9",
        "generated_at": utc_now_compact(),
        "seed": int(args.seed),
        "checks": checks,
        "pass_count": int(passed),
        "total_checks": int(len(checks)),
        "status": status,
        "runtime_seconds": float(time.perf_counter() - started),
    }

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / "stage28_station_safety_audit.md"
    json_path = docs_dir / "stage28_station_safety_audit.json"
    md_path.write_text(_render_md(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"audit_md: {md_path}")
    print(f"audit_json: {json_path}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

