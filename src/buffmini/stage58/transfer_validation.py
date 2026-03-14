"""Stage-58 transfer validation and scope exhaustion decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def assess_transfer_validation(
    *,
    stage57_verdict: str,
    primary_metrics: dict[str, Any],
    transfer_metrics: dict[str, Any] | None,
    transfer_metric_source_type: str = "",
    transfer_artifact_path: str = "",
) -> dict[str, Any]:
    verdict_in = str(stage57_verdict).upper()
    if verdict_in == "STALE_INPUTS":
        return {
            "verdict": "STALE_INPUTS",
            "transfer_acceptable": False,
            "reason": "stage57_stale_inputs",
        }
    if verdict_in == "PARTIAL":
        return {
            "verdict": "PARTIAL",
            "transfer_acceptable": False,
            "reason": "stage57_partial_insufficient_evidence",
        }
    if verdict_in == "NO_EDGE_IN_SCOPE":
        return {
            "verdict": "NO_EDGE_IN_SCOPE",
            "transfer_acceptable": False,
            "reason": "stage57_not_passed",
        }
    if verdict_in != "PASSING_EDGE":
        return {
            "verdict": "PARTIAL",
            "transfer_acceptable": False,
            "reason": "stage57_unknown_verdict",
        }
    source_type = str(transfer_metric_source_type).strip()
    artifact_path = Path(str(transfer_artifact_path).strip()) if str(transfer_artifact_path).strip() else Path("")
    transfer = dict(transfer_metrics or {})
    if not transfer:
        return {
            "verdict": "PARTIAL",
            "transfer_acceptable": False,
            "reason": "missing_transfer_metrics",
            "transfer_source_type": source_type,
            "transfer_artifact_path": str(artifact_path),
        }
    if source_type in {"proxy_only", "synthetic", ""}:
        return {
            "verdict": "PARTIAL",
            "transfer_acceptable": False,
            "reason": "transfer_evidence_not_real",
            "transfer_source_type": source_type,
            "transfer_artifact_path": str(artifact_path),
        }
    if not artifact_path.exists():
        return {
            "verdict": "PARTIAL",
            "transfer_acceptable": False,
            "reason": "missing_transfer_artifact",
            "transfer_source_type": source_type,
            "transfer_artifact_path": str(artifact_path),
        }
    transfer_lcb = float(transfer.get("exp_lcb", 0.0))
    transfer_dd = float(transfer.get("maxDD", transfer.get("max_drawdown", 0.0)))
    primary_lcb = float(primary_metrics.get("exp_lcb", 0.0))
    transfer_acceptable = bool(transfer_lcb > 0.0 and transfer_dd <= 0.25)
    if not transfer_acceptable:
        verdict = "PARTIAL"
        reason = "transfer_not_acceptable"
    elif transfer_lcb >= max(0.0, primary_lcb * 0.75):
        verdict = "MEDIUM_EDGE"
        reason = "transfer_acceptable_medium"
    else:
        verdict = "WEAK_EDGE"
        reason = "transfer_acceptable_weak"
    return {
        "verdict": verdict,
        "transfer_acceptable": transfer_acceptable,
        "reason": reason,
        "primary_exp_lcb": primary_lcb,
        "transfer_exp_lcb": transfer_lcb,
        "transfer_maxDD": transfer_dd,
        "transfer_source_type": source_type,
        "transfer_artifact_path": str(artifact_path),
    }
