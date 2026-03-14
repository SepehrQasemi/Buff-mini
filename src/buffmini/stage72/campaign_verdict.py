"""Stage-72 free campaign verdict logic."""

from __future__ import annotations

from typing import Any

from buffmini.utils.hashing import stable_hash


def derive_free_campaign_verdict(
    *,
    replay_gate: dict[str, Any],
    walkforward_gate: dict[str, Any],
    monte_carlo_gate: dict[str, Any],
    cross_seed_gate: dict[str, Any],
    transfer_acceptable: bool,
    campaign_runs: int,
    frozen_scope_fail_streak: int,
) -> dict[str, Any]:
    replay_pass = bool(replay_gate.get("passed", False))
    wf_pass = bool(walkforward_gate.get("passed", False))
    mc_pass = bool(monte_carlo_gate.get("passed", False))
    cross_pass = bool(cross_seed_gate.get("passed", False))
    all_pass = replay_pass and wf_pass and mc_pass and cross_pass
    if all_pass and transfer_acceptable:
        verdict = "MEDIUM_EDGE"
    elif all_pass:
        verdict = "WEAK_EDGE"
    elif int(frozen_scope_fail_streak) >= 3 and int(campaign_runs) >= 3:
        verdict = "NO_EDGE_IN_SCOPE"
    else:
        verdict = "PARTIAL"
    payload = {
        "verdict": verdict,
        "all_pass": all_pass,
        "transfer_acceptable": bool(transfer_acceptable),
        "campaign_runs": int(campaign_runs),
        "frozen_scope_fail_streak": int(frozen_scope_fail_streak),
        "replay_gate": replay_gate,
        "walkforward_gate": walkforward_gate,
        "monte_carlo_gate": monte_carlo_gate,
        "cross_seed_gate": cross_seed_gate,
    }
    payload["summary_hash"] = stable_hash(payload, length=16)
    return payload

