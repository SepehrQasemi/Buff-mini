"""Stage-55 replay efficiency contracts and budget allocation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buffmini.utils.hashing import stable_hash


REQUIRED_PHASE_TIMINGS: tuple[str, ...] = (
    "candidate_generation",
    "stage_a_gate",
    "stage_b_gate",
    "micro_replay",
    "full_replay",
    "walkforward",
    "monte_carlo",
)


def build_replay_cache_key(
    *,
    data_hash: str,
    setup_signature: str,
    timeframe: str,
    cost_model: str,
    scope_id: str,
) -> str:
    return str(
        stable_hash(
            {
                "data_hash": str(data_hash),
                "setup_signature": str(setup_signature),
                "timeframe": str(timeframe),
                "cost_model": str(cost_model),
                "scope_id": str(scope_id),
            },
            length=24,
        )
    )


def validate_phase_timings(phase_timings: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_PHASE_TIMINGS if key not in phase_timings]
    if missing:
        raise ValueError(f"Missing Stage-55 phase timings: {missing}")
    for key in REQUIRED_PHASE_TIMINGS:
        _ = float(phase_timings[key])


def allocate_replay_budget(
    candidates: pd.DataFrame,
    *,
    budget: dict[str, Any],
) -> dict[str, Any]:
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return {
            "counts": {"input": 0, "precheck": 0, "micro_replay": 0, "full_replay": 0, "walkforward": 0, "monte_carlo": 0},
            "selected_candidate_ids": {"micro_replay": [], "full_replay": [], "walkforward": [], "monte_carlo": []},
        }
    frame["replay_priority"] = pd.to_numeric(frame.get("replay_priority", 0.0), errors="coerce").fillna(0.0)
    ranked = frame.sort_values(["replay_priority", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    precheck_cap = int(max(1, budget.get("setup_generation_cap", budget.get("candidate_limit", len(ranked)))))
    micro_cap = int(max(1, budget.get("micro_replay_cap", budget.get("micro_replay_limit", budget.get("stage_a_cap", budget.get("stage_a_limit", 1))))))
    full_cap = int(max(1, budget.get("full_replay_cap", budget.get("full_replay_limit", budget.get("stage_b_cap", budget.get("stage_b_limit", 1))))))
    wf_cap = int(max(1, budget.get("walkforward_cap", budget.get("walkforward_limit", 1))))
    mc_cap = int(max(0, budget.get("monte_carlo_cap", budget.get("monte_carlo_limit", 1))))
    prechecked = ranked.head(precheck_cap)
    micro = prechecked.head(micro_cap)
    full = micro.head(full_cap)
    wf = full.head(wf_cap)
    mc = wf.head(mc_cap)
    return {
        "counts": {
            "input": int(len(ranked)),
            "precheck": int(len(prechecked)),
            "micro_replay": int(len(micro)),
            "full_replay": int(len(full)),
            "walkforward": int(len(wf)),
            "monte_carlo": int(len(mc)),
        },
        "selected_candidate_ids": {
            "micro_replay": micro.get("candidate_id", pd.Series(dtype=str)).astype(str).tolist(),
            "full_replay": full.get("candidate_id", pd.Series(dtype=str)).astype(str).tolist(),
            "walkforward": wf.get("candidate_id", pd.Series(dtype=str)).astype(str).tolist(),
            "monte_carlo": mc.get("candidate_id", pd.Series(dtype=str)).astype(str).tolist(),
        },
    }


def estimate_replay_speedup(*, baseline_runtime_seconds: float, optimized_runtime_seconds: float) -> dict[str, Any]:
    baseline = float(max(baseline_runtime_seconds, 1e-9))
    optimized = float(max(optimized_runtime_seconds, 0.0))
    improvement_pct = float(((baseline - optimized) / baseline) * 100.0)
    return {
        "baseline_runtime_seconds": baseline,
        "optimized_runtime_seconds": optimized,
        "improvement_pct": float(round(improvement_pct, 6)),
        "meets_stage55_target": bool(improvement_pct >= 40.0),
    }
