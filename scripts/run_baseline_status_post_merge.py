from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Write baseline truth after merged main verification")
    parser.add_argument("--docs-dir", default="docs")
    args = parser.parse_args()

    docs_dir = ROOT / args.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    current_branch = _run(["git", "branch", "--show-current"])
    local_head = _run(["git", "rev-parse", "HEAD"])
    main_head = _run(["git", "rev-parse", "main"])
    origin_main = _run(["git", "rev-parse", "origin/main"])
    worktree_clean = len(_run(["git", "status", "--porcelain"]).strip()) == 0
    status_short = _run(["git", "status", "--short", "--branch"])

    open_prs = _maybe_json(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--json",
            "number,title,headRefName,baseRefName,url,isDraft,mergeStateStatus",
        ]
    )
    protection = _maybe_json(["gh", "api", "repos/SepehrQasemi/Buff-mini/branches/main/protection"]) or {}

    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_accuracy_checks = {
        "rewritten_banner_present": "Buff-mini is a local, evidence-disciplined crypto research engine" in readme_text,
        "run_modes_present": "Run Modes" in readme_text,
        "validation_philosophy_present": "Validation Philosophy" in readme_text,
        "stage_95_103_present": "Stage-95" in readme_text or "Stage-95 through Stage-103" in readme_text,
    }

    master_summary_path = docs_dir / "master_execution_summary.json"
    prior_master = json.loads(master_summary_path.read_text(encoding="utf-8")) if master_summary_path.exists() else {}
    known_blockers = []
    campaign_outcome = dict(prior_master.get("campaign_outcome", {}))
    classification = str(campaign_outcome.get("classification", "")).strip()
    if classification:
        known_blockers.append(classification)
    if not all(bool(value) for value in readme_accuracy_checks.values() if isinstance(value, bool)):
        known_blockers.append("README_NEEDS_STAGE95_103_UPDATE")

    baseline_readme_accurate = bool(
        readme_accuracy_checks["rewritten_banner_present"]
        and readme_accuracy_checks["run_modes_present"]
        and readme_accuracy_checks["validation_philosophy_present"]
    )

    payload = {
        "baseline_branch": "main",
        "baseline_commit": main_head,
        "current_branch": current_branch,
        "current_head": local_head,
        "origin_main_commit": origin_main,
        "local_main_matches_origin_main": bool(main_head == origin_main),
        "working_tree_clean": bool(worktree_clean),
        "git_status_short": status_short,
        "open_pr_inventory": open_prs if isinstance(open_prs, list) else [],
        "branch_protection": {
            "required_pull_request_reviews": bool(protection.get("required_pull_request_reviews")),
            "required_approving_review_count": (protection.get("required_pull_request_reviews") or {}).get("required_approving_review_count", 0),
            "allow_force_pushes": (protection.get("allow_force_pushes") or {}).get("enabled", False),
            "allow_deletions": (protection.get("allow_deletions") or {}).get("enabled", False),
        },
        "readme_accuracy": readme_accuracy_checks,
        "baseline_readme_accurate": baseline_readme_accurate,
        "current_merged_baseline_valid_for_stage95": bool(main_head == origin_main and baseline_readme_accurate),
        "known_blockers_from_latest_merged_state": known_blockers,
    }

    (docs_dir / "baseline_status_post_merge.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Baseline Status Post Merge Report",
        "",
        "## Baseline",
        f"- baseline_branch: `{payload['baseline_branch']}`",
        f"- baseline_commit: `{payload['baseline_commit']}`",
        f"- current_branch: `{payload['current_branch']}`",
        f"- current_head: `{payload['current_head']}`",
        f"- origin_main_commit: `{payload['origin_main_commit']}`",
        f"- local_main_matches_origin_main: `{payload['local_main_matches_origin_main']}`",
        f"- working_tree_clean: `{payload['working_tree_clean']}`",
        f"- baseline_readme_accurate: `{payload['baseline_readme_accurate']}`",
        "",
        "## Protection",
        f"- pull_request_reviews_required: `{payload['branch_protection']['required_pull_request_reviews']}`",
        f"- required_approving_review_count: `{payload['branch_protection']['required_approving_review_count']}`",
        f"- allow_force_pushes: `{payload['branch_protection']['allow_force_pushes']}`",
        f"- allow_deletions: `{payload['branch_protection']['allow_deletions']}`",
        "",
        "## README Accuracy",
    ]
    for key, value in payload["readme_accuracy"].items():
        report_lines.append(f"- {key}: `{value}`")
    report_lines.extend(
        [
            "",
            "## Known Blockers From Latest Merged State",
            *([f"- `{item}`" for item in payload["known_blockers_from_latest_merged_state"]] or ["- none"]),
            "",
            f"- current_merged_baseline_valid_for_stage95: `{payload['current_merged_baseline_valid_for_stage95']}`",
        ]
    )
    (docs_dir / "baseline_status_post_merge_report.md").write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
