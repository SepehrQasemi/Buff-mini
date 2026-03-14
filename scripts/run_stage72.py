"""Run Stage-72 free campaign verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage72 import derive_free_campaign_verdict
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-72 free campaign verdict")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--campaign-runs", type=int, default=20)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage57 = _load_json(docs_dir / "stage57_summary.json")
    stage58 = _load_json(docs_dir / "stage58_summary.json")
    history_path = docs_dir / "stage72_campaign_history.json"
    history = []
    if history_path.exists():
        loaded = json.loads(history_path.read_text(encoding="utf-8"))
        history = loaded if isinstance(loaded, list) else []
    fail_streak = 0
    for rec in reversed(history):
        verdict = str(rec.get("verdict", "")).upper()
        if verdict not in {"WEAK_EDGE", "MEDIUM_EDGE", "PASSING_EDGE"}:
            fail_streak += 1
        else:
            break
    transfer_ok = bool(stage58.get("transfer_result", {}).get("transfer_acceptable", False))
    decision_evidence_ok = bool(stage57.get("decision_evidence", {}).get("allowed", False))
    payload = derive_free_campaign_verdict(
        replay_gate=dict(stage57.get("replay_gate", {})),
        walkforward_gate=dict(stage57.get("walkforward_gate", {})),
        monte_carlo_gate=dict(stage57.get("monte_carlo_gate", {})),
        cross_seed_gate=dict(stage57.get("cross_seed_gate", {})),
        transfer_acceptable=transfer_ok,
        campaign_runs=int(max(1, args.campaign_runs)),
        frozen_scope_fail_streak=int(fail_streak),
    )
    if not decision_evidence_ok:
        payload["verdict"] = "PARTIAL"
        payload["all_pass"] = False
    history.append(
        {
            "campaign_id": stable_hash({"n": len(history), "runs": int(args.campaign_runs)}, length=12),
            "verdict": payload["verdict"],
            "summary_hash": payload["summary_hash"],
        }
    )
    history_path.write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
    summary = {
        "stage": "72",
        "status": "SUCCESS" if payload["verdict"] in {"WEAK_EDGE", "MEDIUM_EDGE", "NO_EDGE_IN_SCOPE"} and decision_evidence_ok else "PARTIAL",
        "execution_status": "EXECUTED",
        "stage_role": "real_validation",
        "validation_state": "FINAL_VERDICT_WITH_REAL_EVIDENCE" if decision_evidence_ok else "FINAL_VERDICT_BLOCKED_EVIDENCE_INSUFFICIENT",
        "campaign_runs": int(max(1, args.campaign_runs)),
        "frozen_scope_fail_streak": int(fail_streak),
        "verdict": payload["verdict"],
        "decision_evidence_allowed": decision_evidence_ok,
        "transfer_acceptable": bool(payload["transfer_acceptable"]),
        "all_pass": bool(payload["all_pass"]),
        "summary_hash": stable_hash(
            {"payload_hash": payload["summary_hash"], "history_len": len(history)},
            length=16,
        ),
    }
    (docs_dir / "stage72_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage72_report.md").write_text(
        "\n".join(
            [
                "# Stage-72 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- stage_role: `{summary['stage_role']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- campaign_runs: `{summary['campaign_runs']}`",
                f"- frozen_scope_fail_streak: `{summary['frozen_scope_fail_streak']}`",
                f"- verdict: `{summary['verdict']}`",
                f"- decision_evidence_allowed: `{summary['decision_evidence_allowed']}`",
                f"- transfer_acceptable: `{summary['transfer_acceptable']}`",
                f"- all_pass: `{summary['all_pass']}`",
                f"- summary_hash: `{summary['summary_hash']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
