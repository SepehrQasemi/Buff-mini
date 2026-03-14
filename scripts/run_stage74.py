"""Generate Stage-74 summary/report artifacts from the repaired repo state."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage-74 report artifacts")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--compileall-status", type=str, default="NOT_VERIFIED")
    parser.add_argument("--pytest-status", type=str, default="NOT_VERIFIED")
    parser.add_argument("--integration-status", type=str, default="NOT_VERIFIED")
    parser.add_argument("--push-status", type=str, default="NOT_ATTEMPTED")
    parser.add_argument("--pr-url", type=str, default="")
    parser.add_argument("--pr-status", type=str, default="NOT_ATTEMPTED")
    parser.add_argument("--protection-status", type=str, default="NOT_ATTEMPTED")
    parser.add_argument("--protection-detail", type=str, default="")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return str(result.stdout).strip()


def _final_verdict(*, compileall_status: str, pytest_status: str, integration_status: str, pr_status: str, protection_status: str) -> str:
    compile_ok = str(compileall_status).upper() == "PASS"
    pytest_ok = str(pytest_status).upper() == "PASS"
    integration_ok = str(integration_status).upper() == "PASS"
    pr_ok = str(pr_status).upper() == "OPEN"
    protection_ok = str(protection_status).upper() == "PASS"
    if compile_ok and pytest_ok and integration_ok and pr_ok and protection_ok:
        return "FULL_REPAIR_SUBSTANTIAL"
    if compile_ok and pytest_ok and integration_ok and pr_ok:
        return "MAJOR_REPAIR_MOSTLY_COMPLETE"
    if compile_ok and pytest_ok and integration_ok:
        return "PARTIAL_REPAIR_MEANINGFUL"
    return "LIMITED_REPAIR_ONLY"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    runs_dir = Path(args.runs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    stage53 = _load_json(docs_dir / "stage53_summary.json")
    stage57 = _load_json(docs_dir / "stage57_summary.json")
    stage58 = _load_json(docs_dir / "stage58_summary.json")
    stage61 = _load_json(docs_dir / "stage61_summary.json")
    stage67 = _load_json(docs_dir / "stage67_summary.json")
    stage72 = _load_json(docs_dir / "stage72_summary.json")
    full_trace = _load_json(docs_dir / "full_trace_summary.json")

    branch_name = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    commit_hash = _git_output("rev-parse", "HEAD")
    changed_files = [line for line in _git_output("diff", "--name-only", "origin/main...HEAD").splitlines() if line.strip()]
    if not changed_files:
        changed_files = [line for line in _git_output("diff", "--name-only", "HEAD").splitlines() if line.strip()]

    verdict = _final_verdict(
        compileall_status=str(args.compileall_status),
        pytest_status=str(args.pytest_status),
        integration_status=str(args.integration_status),
        pr_status=str(args.pr_status),
        protection_status=str(args.protection_status),
    )

    summary = {
        "stage": "74",
        "status": "SUCCESS",
        "branch_name": branch_name,
        "commit_hash": commit_hash,
        "compileall_status": str(args.compileall_status),
        "pytest_status": str(args.pytest_status),
        "integration_status": str(args.integration_status),
        "push_status": str(args.push_status),
        "pr_url": str(args.pr_url),
        "pr_status": str(args.pr_status),
        "protection_status": str(args.protection_status),
        "protection_detail": str(args.protection_detail),
        "changed_files": changed_files,
        "highlights": {
            "stage53": {
                "status": stage53.get("status"),
                "execution_status": stage53.get("execution_status"),
                "validated_candidate_id": stage53.get("validated_candidate_id"),
                "replay_metrics_artifact_path": stage53.get("replay_metrics_artifact_path"),
            },
            "stage57": {
                "status": stage57.get("status"),
                "validation_state": stage57.get("validation_state"),
                "verdict": stage57.get("verdict"),
                "decision_evidence_allowed": (stage57.get("decision_evidence", {}) or {}).get("allowed"),
            },
            "stage58": {
                "status": stage58.get("status"),
                "validation_state": stage58.get("validation_state"),
                "transfer_artifact_exists": stage58.get("transfer_artifact_exists"),
                "evidence_quality": stage58.get("evidence_quality"),
            },
            "stage61": {
                "status": stage61.get("status"),
                "decision_evidence_allowed": stage61.get("decision_evidence_allowed"),
            },
            "stage67": {
                "status": stage67.get("status"),
                "validation_state": stage67.get("validation_state"),
                "continuity_blocked": stage67.get("continuity_blocked"),
                "walkforward_artifact_path": stage67.get("walkforward_artifact_path"),
            },
            "stage72": {
                "status": stage72.get("status"),
                "verdict": stage72.get("verdict"),
            },
            "evidence_quality": full_trace.get("evidence_quality", {}),
        },
        "final_verdict": verdict,
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage74_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    lines = [
        "# Stage-74 Summary Report",
        "",
        f"- branch_name: `{branch_name}`",
        f"- commit_hash: `{commit_hash}`",
        f"- compileall_status: `{args.compileall_status}`",
        f"- pytest_status: `{args.pytest_status}`",
        f"- integration_status: `{args.integration_status}`",
        f"- pr_status: `{args.pr_status}`",
        f"- pr_url: `{args.pr_url}`",
        f"- protection_status: `{args.protection_status}`",
        f"- protection_detail: `{args.protection_detail}`",
        "",
        "## Runtime Highlights",
        f"- stage53: `{summary['highlights']['stage53']}`",
        f"- stage57: `{summary['highlights']['stage57']}`",
        f"- stage58: `{summary['highlights']['stage58']}`",
        f"- stage61: `{summary['highlights']['stage61']}`",
        f"- stage67: `{summary['highlights']['stage67']}`",
        f"- stage72: `{summary['highlights']['stage72']}`",
        f"- evidence_quality: `{summary['highlights']['evidence_quality']}`",
        "",
        "## Changed Files",
    ]
    lines.extend([f"- `{path}`" for path in changed_files] or ["- `(none detected)`"])
    lines.extend(["", "## Final Verdict", f"`{verdict}`", "", f"- summary_hash: `{summary['summary_hash']}`"])
    report_text = "\n".join(lines) + "\n"
    (docs_dir / "stage74_report.md").write_text(report_text, encoding="utf-8")

    run_id = str(stage67.get("stage28_run_id", "")).strip()
    if run_id:
        out_dir = runs_dir / run_id / "stage74"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        (out_dir / "report.md").write_text(report_text, encoding="utf-8")

    print(f"status: {summary['status']}")
    print(f"final_verdict: {verdict}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
