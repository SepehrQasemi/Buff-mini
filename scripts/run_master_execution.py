from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
STAGES = tuple(range(95, 104))


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
    payload = _maybe_json(
        [
            "gh",
            "pr",
            "list",
            "--head",
            branch,
            "--state",
            "open",
            "--json",
            "number,title,url,state,headRefName,baseRefName,mergeStateStatus,isDraft,statusCheckRollup",
        ]
    )
    if isinstance(payload, list) and payload:
        item = payload[0]
        if isinstance(item, dict):
            return item
    return {}


def build_master_summary(branch: str, head_commit: str) -> dict[str, Any]:
    baseline = _read_json(DOCS / "baseline_status_post_merge.json")
    stage_payloads: dict[str, dict[str, Any]] = {}
    stage_statuses: dict[str, Any] = {}
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
    stage95 = stage_payloads.get("95", {})
    stage96 = stage_payloads.get("96", {})
    stage100 = stage_payloads.get("100", {})
    stage101 = stage_payloads.get("101", {})
    stage102 = stage_payloads.get("102", {})
    stage103 = stage_payloads.get("103", {})

    pr = _current_pr_for_branch(branch)
    protection = _maybe_json(["gh", "api", "repos/SepehrQasemi/Buff-mini/branches/main/protection"]) or {}

    if str(stage103.get("final_edge_verdict", "")) == "ROBUST_CANDIDATE_FOUND":
        final_verdict = "FULL_REPAIR_SUBSTANTIAL"
    elif str(stage103.get("final_edge_verdict", "")) in {
        "PROMISING_BUT_UNPROVEN_CANDIDATES_FOUND",
        "WEAK_REGIME_LOCAL_MECHANISMS_FOUND",
    }:
        final_verdict = "MAJOR_REPAIR_MOSTLY_COMPLETE"
    elif str(stage103.get("final_edge_verdict", "")) in {
        "GENERATOR_OR_SEARCH_FORMALISM_STILL_INSUFFICIENT",
        "DATA_OR_SCOPE_BLOCKS_STRONGER_CONCLUSION",
    }:
        final_verdict = "PARTIAL_REPAIR_MEANINGFUL"
    else:
        final_verdict = "LIMITED_REPAIR_ONLY"

    summary = {
        "execution_branch": branch,
        "execution_head": head_commit,
        "baseline": {
            "branch": baseline.get("baseline_branch", "main"),
            "commit": baseline.get("baseline_commit", ""),
            "local_main_matches_origin_main": baseline.get("local_main_matches_origin_main", False),
            "baseline_readme_accurate": baseline.get("baseline_readme_accurate", False),
        },
        "stage_statuses": stage_statuses,
        "controlled_detectability": {
            "proven": bool(stage76.get("status") == "SUCCESS"),
            "signal_detection_rate": stage76.get("signal_detection_rate"),
            "bad_control_rejection_rate": stage76.get("bad_control_rejection_rate"),
            "synthetic_winner_recall": stage76.get("synthetic_winner_recall"),
            "false_negative_rate_on_known_good": stage76.get("false_negative_rate_on_known_good"),
        },
        "stage95_usefulness": {
            "stage95b_recommended": stage95.get("stage95b_recommended"),
            "stage95b_applied": stage95.get("stage95b_applied"),
            "dead_weight_family_count": len(stage95.get("dead_weight_families", [])),
            "usefulness_delta": stage95.get("usefulness_delta", {}),
        },
        "stage96_canonical": {
            "snapshot_id": stage96.get("snapshot_id", ""),
            "snapshot_hash": stage96.get("snapshot_hash", ""),
            "strict_usable_rows": int(sum(1 for row in stage96.get("repair_rows", []) if bool(row.get("strict_usable_after", False)))),
        },
        "stage100_truth_campaign": {
            "truth_counts": stage100.get("truth_counts", {}),
            "tier1_symbols": stage100.get("tier1_symbols", []),
            "tier2_symbols": stage100.get("tier2_symbols", []),
            "candidate_limit_per_scope": stage100.get("candidate_limit_per_scope"),
        },
        "stage101_null_attack": {
            "candidate_count_reviewed": stage101.get("candidate_count_reviewed", 0),
            "control_win_counts": stage101.get("control_win_counts", {}),
            "candidate_beats_all_controls_count": stage101.get("candidate_beats_all_controls_count", 0),
            "candidate_beats_majority_controls_count": stage101.get("candidate_beats_majority_controls_count", 0),
        },
        "stage102_rescue": {
            "candidate_limit_reviewed": stage102.get("candidate_limit_reviewed", 0),
            "classification_counts": stage102.get("classification_counts", {}),
        },
        "stage103_final_edge_verdict": {
            "final_edge_verdict": stage103.get("final_edge_verdict", ""),
            "evidence_table": stage103.get("evidence_table", []),
        },
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
    detectability = master.get("controlled_detectability", {})
    stage95 = master.get("stage95_usefulness", {})
    stage96 = master.get("stage96_canonical", {})
    stage100 = master.get("stage100_truth_campaign", {})
    stage101 = master.get("stage101_null_attack", {})
    stage102 = master.get("stage102_rescue", {})
    stage103 = master.get("stage103_final_edge_verdict", {})
    pr = master.get("stage_pr", {})
    protection = master.get("main_branch_protection", {})
    return f"""# Master Execution Report

## Baseline
- baseline_branch: `{master['baseline']['branch']}`
- baseline_commit: `{master['baseline']['commit']}`
- local_main_matches_origin_main: `{master['baseline']['local_main_matches_origin_main']}`
- baseline_readme_accurate: `{master['baseline']['baseline_readme_accurate']}`

## Stage Statuses
{chr(10).join(stage_lines)}

## Controlled Detectability
- proven: `{detectability.get('proven')}`
- signal_detection_rate: `{detectability.get('signal_detection_rate')}`
- bad_control_rejection_rate: `{detectability.get('bad_control_rejection_rate')}`
- synthetic_winner_recall: `{detectability.get('synthetic_winner_recall')}`
- false_negative_rate_on_known_good: `{detectability.get('false_negative_rate_on_known_good')}`

## Stage-95 Usefulness
- stage95b_recommended: `{stage95.get('stage95b_recommended')}`
- stage95b_applied: `{stage95.get('stage95b_applied')}`
- dead_weight_family_count: `{stage95.get('dead_weight_family_count')}`
- usefulness_delta: `{stage95.get('usefulness_delta')}`

## Stage-96 Canonical
- snapshot_id: `{stage96.get('snapshot_id')}`
- snapshot_hash: `{stage96.get('snapshot_hash')}`
- strict_usable_rows: `{stage96.get('strict_usable_rows')}`

## Stage-100 Truth Campaign
- truth_counts: `{stage100.get('truth_counts')}`
- tier1_symbols: `{stage100.get('tier1_symbols')}`
- tier2_symbols: `{stage100.get('tier2_symbols')}`
- candidate_limit_per_scope: `{stage100.get('candidate_limit_per_scope')}`

## Stage-101 Null Attack
- candidate_count_reviewed: `{stage101.get('candidate_count_reviewed')}`
- control_win_counts: `{stage101.get('control_win_counts')}`
- candidate_beats_all_controls_count: `{stage101.get('candidate_beats_all_controls_count')}`
- candidate_beats_majority_controls_count: `{stage101.get('candidate_beats_majority_controls_count')}`

## Stage-102 Rescue
- candidate_limit_reviewed: `{stage102.get('candidate_limit_reviewed')}`
- classification_counts: `{stage102.get('classification_counts')}`

## Stage-103 Final Edge Verdict
- final_edge_verdict: `{stage103.get('final_edge_verdict')}`
- evidence_table: `{stage103.get('evidence_table')}`

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
"""


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
