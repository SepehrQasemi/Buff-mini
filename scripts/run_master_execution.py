from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
STAGES = tuple(range(85, 95))


def _run(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _maybe_json(command: list[str]) -> dict[str, Any] | list[Any] | None:
    try:
        raw = _run(command)
    except Exception:
        return None
    if not raw:
        return None
    return json.loads(raw)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _current_pr_for_branch(branch: str) -> dict[str, Any]:
    payload = _maybe_json([
        "gh", "pr", "list", "--head", branch, "--state", "open", "--json", "number,title,url,state,headRefName,baseRefName,mergeStateStatus,isDraft,statusCheckRollup"
    ])
    if isinstance(payload, list) and payload:
        item = payload[0]
        if isinstance(item, dict):
            return item
    return {}


def build_master_summary(branch: str, head_commit: str) -> dict[str, Any]:
    baseline = _read_json(DOCS / "baseline_status.json")
    prereq = _read_json(DOCS / "prereq_finalize_summary.json")
    stage_statuses: dict[str, Any] = {}
    stage_payloads: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        path = DOCS / f"stage{stage}_summary.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        stage_payloads[str(stage)] = payload
        stage_statuses[str(stage)] = {
            "status": payload.get("status", ""),
            "execution_status": payload.get("execution_status", ""),
            "validation_state": payload.get("validation_state", ""),
            "summary_hash": payload.get("summary_hash", ""),
        }

    stage76 = _read_json(DOCS / "stage76_summary.json") if (DOCS / "stage76_summary.json").exists() else {}
    stage89 = stage_payloads.get("89", {})
    stage94 = stage_payloads.get("94", {})
    stage90 = stage_payloads.get("90", {})
    pr = _current_pr_for_branch(branch)
    protection = _maybe_json(["gh", "api", "repos/SepehrQasemi/Buff-mini/branches/main/protection"]) or {}

    detectability = {
        "signal_detection_rate": stage76.get("signal_detection_rate"),
        "bad_control_rejection_rate": stage76.get("bad_control_rejection_rate"),
        "synthetic_winner_recall": stage76.get("synthetic_winner_recall"),
        "false_negative_rate_on_known_good": stage76.get("false_negative_rate_on_known_good"),
        "proven": bool(stage76.get("status") == "SUCCESS"),
    }
    data_fitness = {
        "evaluation_usable_class_counts": stage89.get("evaluation_usable_class_counts", {}),
        "snapshot_meta": stage89.get("snapshot_meta", {}),
    }
    campaign_outcome = {
        "campaign_outcome": stage94.get("campaign_outcome", ""),
        "classification": stage94.get("classification", ""),
        "candidate_class_counts": stage94.get("candidate_class_counts", {}),
        "blocked_scope_rows": len(stage94.get("blocked_scope_rows", [])),
    }

    if stage94.get("classification") == "system_blocked_uninterpretable":
        final_verdict = "MAJOR_REPAIR_MOSTLY_COMPLETE"
    elif any((stage_payloads.get(str(stage), {}) or {}).get("status") != "SUCCESS" for stage in STAGES):
        final_verdict = "PARTIAL_REPAIR_MEANINGFUL"
    else:
        final_verdict = "FULL_REPAIR_SUBSTANTIAL"

    summary = {
        "execution_branch": branch,
        "execution_head": head_commit,
        "baseline": {
            "branch": baseline.get("current_effective_baseline", {}).get("branch", "main"),
            "commit": baseline.get("current_effective_baseline", {}).get("commit", ""),
            "baseline_type": baseline.get("current_effective_baseline", {}).get("baseline_type", "merged_main"),
        },
        "prereq": {
            "status": prereq.get("status", ""),
            "validation_state": prereq.get("validation_state", ""),
            "summary_hash": prereq.get("summary_hash", ""),
        },
        "stage_statuses": stage_statuses,
        "controlled_detectability": detectability,
        "data_fitness": data_fitness,
        "campaign_outcome": campaign_outcome,
        "stage90_dominant_culprit": stage90.get("dominant_culprit", ""),
        "stage_pr": {
            "number": pr.get("number"),
            "title": pr.get("title", ""),
            "url": pr.get("url", ""),
            "state": pr.get("state", ""),
            "head": pr.get("headRefName", ""),
            "base": pr.get("baseRefName", ""),
            "merge_state_status": pr.get("mergeStateStatus", ""),
            "is_draft": bool(pr.get("isDraft", False)),
            "checks": [
                {
                    "name": item.get("name", ""),
                    "status": item.get("status", ""),
                    "conclusion": item.get("conclusion", ""),
                }
                for item in (pr.get("statusCheckRollup") or [])
            ],
        },
        "main_branch_protection": {
            "required_pull_request_reviews": bool(protection.get("required_pull_request_reviews")),
            "required_approving_review_count": (protection.get("required_pull_request_reviews") or {}).get("required_approving_review_count", 0),
            "allow_force_pushes": (protection.get("allow_force_pushes") or {}).get("enabled", False),
            "allow_deletions": (protection.get("allow_deletions") or {}).get("enabled", False),
        },
        "final_verdict": final_verdict,
    }
    summary["summary_hash"] = _hash_payload(summary)
    return summary


def build_master_report(master: dict[str, Any]) -> str:
    stage_lines = []
    for stage in STAGES:
        payload = master.get("stage_statuses", {}).get(str(stage), {})
        stage_lines.append(
            f"- Stage-{stage}: status=`{payload.get('status', '')}`, execution_status=`{payload.get('execution_status', '')}`, validation_state=`{payload.get('validation_state', '')}`, summary_hash=`{payload.get('summary_hash', '')}`"
        )
    pr = master.get("stage_pr", {})
    protection = master.get("main_branch_protection", {})
    detectability = master.get("controlled_detectability", {})
    data_fitness = master.get("data_fitness", {})
    campaign_outcome = master.get("campaign_outcome", {})
    return f'''# Master Execution Report

## Baseline And Prerequisites
- baseline_branch: `{master['baseline']['branch']}`
- baseline_commit: `{master['baseline']['commit']}`
- baseline_type: `{master['baseline']['baseline_type']}`
- prereq_status: `{master['prereq']['status']}`
- prereq_validation_state: `{master['prereq']['validation_state']}`
- prereq_summary_hash: `{master['prereq']['summary_hash']}`

## Stage-85 Through Stage-94
{chr(10).join(stage_lines)}

## Controlled Detectability
- proven: `{detectability.get('proven')}`
- signal_detection_rate: `{detectability.get('signal_detection_rate')}`
- bad_control_rejection_rate: `{detectability.get('bad_control_rejection_rate')}`
- synthetic_winner_recall: `{detectability.get('synthetic_winner_recall')}`
- false_negative_rate_on_known_good: `{detectability.get('false_negative_rate_on_known_good')}`

## Data Fitness
- evaluation_usable_class_counts: `{data_fitness.get('evaluation_usable_class_counts')}`
- snapshot_meta: `{data_fitness.get('snapshot_meta')}`

## Campaign Outcome
- campaign_outcome: `{campaign_outcome.get('campaign_outcome')}`
- classification: `{campaign_outcome.get('classification')}`
- candidate_class_counts: `{campaign_outcome.get('candidate_class_counts')}`
- blocked_scope_rows: `{campaign_outcome.get('blocked_scope_rows')}`
- stage90_dominant_culprit: `{master.get('stage90_dominant_culprit', '')}`

## GitHub
- execution_branch: `{master['execution_branch']}`
- execution_head: `{master['execution_head']}`
- PR number: `{pr.get('number')}`
- PR title: `{pr.get('title')}`
- PR url: `{pr.get('url')}`
- PR state: `{pr.get('state')}`
- PR merge_state_status: `{pr.get('merge_state_status')}`

## Main Protection
- required_pull_request_reviews: `{protection.get('required_pull_request_reviews')}`
- required_approving_review_count: `{protection.get('required_approving_review_count')}`
- allow_force_pushes: `{protection.get('allow_force_pushes')}`
- allow_deletions: `{protection.get('allow_deletions')}`

## Final Verdict
- final_verdict: `{master['final_verdict']}`
- summary_hash: `{master['summary_hash']}`
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default="docs")
    args = parser.parse_args()

    docs_dir = ROOT / args.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    branch = _run(["git", "branch", "--show-current"])
    head_commit = _run(["git", "rev-parse", "HEAD"])
    master = build_master_summary(branch=branch, head_commit=head_commit)
    _write_json(docs_dir / "master_execution_summary.json", master)
    (docs_dir / "master_execution_report.md").write_text(build_master_report(master), encoding="utf-8")


if __name__ == "__main__":
    main()
