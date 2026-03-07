"""Stage-49 self-learning 3.0 memory and mutation logic."""

from __future__ import annotations

from typing import Any

import pandas as pd


def expand_registry_rows_v3(
    *,
    base_rows: list[dict[str, Any]],
    seed: int,
    run_id: str,
    stage47_counts: dict[str, int],
    stage48: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand prior registry rows into Stage-49 v3 schema."""

    out: list[dict[str, Any]] = []
    stage_a = int(stage48.get("stage_a_survivors_after", 0))
    stage_b = int(stage48.get("stage_b_survivors_after", 0))
    shortlisted = int(stage48.get("strict_direct_survivors_before", 0))
    for family, count in sorted(stage47_counts.items(), key=lambda kv: str(kv[0])):
        motif = "NO_RAW_SIGNAL" if int(count) <= 0 else ("REJECT::COST_DRAG" if stage_b <= 0 else "REJECT::NO_SIGNAL")
        row = {
            "run_id": str(run_id),
            "seed": int(seed),
            "module_name": "stage47_signal_genesis2",
            "family_name": str(family),
            "setup_name": f"{family}_setup",
            "context_name": "mixed_context",
            "feature_subset_signature": f"family::{family}",
            "threshold_config": {
                "stage_a_threshold": float(stage48.get("stage_a_threshold", 0.42)),
                "stage_b_threshold": float(stage48.get("stage_b_threshold", 0.0)),
            },
            "raw_candidate_count": int(count),
            "shortlisted_count": int(shortlisted),
            "activation_rate": float(stage_b / max(1, stage_a)),
            "final_trade_count": float(stage_b),
            "exp_lcb": float(stage48.get("net_return_after_cost_mean", 0.0)),
            "top_reject_reason": "cost_drag" if motif == "REJECT::COST_DRAG" else "no_signal",
            "failure_motif_tags": [motif],
            "elite": False,
            "mutation_origin": "stage49_upgrade",
            "runtime_cost": float(stage48.get("runtime_cost", 0.0)),
            "contribution_summary": {
                "candidate_share": float(count / max(1, sum(stage47_counts.values()))),
                "stage_a_survival_lift": float(stage_a / max(1, shortlisted)),
                "stage_b_survival_lift": float(stage_b / max(1, stage_a)),
            },
        }
        out.append(row)

    if not out:
        for row in base_rows:
            item = dict(row)
            item.setdefault("run_id", str(run_id))
            item.setdefault("seed", int(seed))
            item.setdefault("module_name", "stage49_fallback")
            item.setdefault("family_name", str(item.get("family", "__all__")))
            item.setdefault("setup_name", "fallback_setup")
            item.setdefault("context_name", "global")
            item.setdefault("feature_subset_signature", "family::__all__")
            item.setdefault("threshold_config", {})
            item.setdefault("raw_candidate_count", int(item.get("raw_signal_count", 0)))
            item.setdefault("shortlisted_count", int(shortlisted))
            item.setdefault("activation_rate", float(item.get("activation_rate", 0.0)))
            item.setdefault("final_trade_count", float(item.get("final_trade_count", 0.0)))
            item.setdefault("exp_lcb", float(item.get("exp_lcb", 0.0)))
            item.setdefault("top_reject_reason", str(item.get("top_reject_reason", "no_signal")))
            item.setdefault("failure_motif_tags", list(item.get("failure_motif_tags", ["NO_RAW_SIGNAL"])))
            item.setdefault("elite", bool(item.get("elite", False)))
            item.setdefault("mutation_origin", "stage49_fallback")
            item.setdefault("runtime_cost", 0.0)
            item.setdefault("contribution_summary", {"candidate_share": 0.0, "stage_a_survival_lift": 0.0, "stage_b_survival_lift": 0.0})
            out.append(item)
    return out


def failure_aware_mutation(row: dict[str, Any]) -> str:
    """Return mutation guidance from failure motifs and reject reason."""

    motifs = [str(m) for m in row.get("failure_motif_tags", [])]
    top_reject = str(row.get("top_reject_reason", "")).lower()
    raw = int(row.get("raw_candidate_count", 0))
    if raw <= 0 or "NO_RAW_SIGNAL" in motifs:
        return "widen_context_and_expand_grammar"
    if "REJECT::NO_SIGNAL" in motifs:
        return "reshape_trigger_composition"
    if "REJECT::BAD_RR" in motifs:
        return "alter_geometry_and_invalidation"
    if "REJECT::COST_DRAG" in motifs or top_reject == "cost_drag":
        return "prioritize_high_edge_per_trade_setups"
    if "REJECT::WEAK_FLOW_CONTEXT" in motifs:
        return "strengthen_flow_confirmation"
    if "REJECT::FAILED_LIQUIDITY_CONFIRMATION" in motifs:
        return "tighten_liquidity_confirmation"
    if "REJECT::FEASIBILITY_FAIL" in motifs:
        return "adjust_execution_and_sizing_filters"
    return "explore_local_variants"


def deterministic_elites(rows: list[dict[str, Any]], *, top_k: int = 5) -> list[dict[str, Any]]:
    """Select elites deterministically."""

    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    frame["exp_lcb"] = pd.to_numeric(frame.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
    frame["activation_rate"] = pd.to_numeric(frame.get("activation_rate", 0.0), errors="coerce").fillna(0.0)
    frame["raw_candidate_count"] = pd.to_numeric(frame.get("raw_candidate_count", 0), errors="coerce").fillna(0).astype(int)
    frame["family_name"] = frame.get("family_name", "").astype(str)
    frame = frame.sort_values(
        ["exp_lcb", "activation_rate", "raw_candidate_count", "family_name"],
        ascending=[False, False, False, True],
    ).head(int(max(1, top_k)))
    elites = frame.to_dict(orient="records")
    for row in elites:
        row["elite"] = True
    return elites


def family_module_downweighting(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Downweight weak families/modules based on repeated motifs and low contributions."""

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    frame["family_name"] = frame.get("family_name", "__all__").astype(str)
    frame["exp_lcb"] = pd.to_numeric(frame.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
    frame["activation_rate"] = pd.to_numeric(frame.get("activation_rate", 0.0), errors="coerce").fillna(0.0)
    weights: dict[str, float] = {}
    for family, grp in frame.groupby("family_name", dropna=False):
        mean_lcb = float(grp["exp_lcb"].mean())
        mean_activation = float(grp["activation_rate"].mean())
        penalty = 0.2 if mean_lcb <= 0.0 else 0.0
        base = max(0.1, min(1.5, mean_activation + 0.5 - penalty))
        weights[str(family)] = float(round(base, 6))
    return weights


def learning_depth_assessment(rows: list[dict[str, Any]], *, family_weights: dict[str, float]) -> str:
    """Assess Stage-49 learning depth level."""

    if not rows:
        return "SHALLOW_NO_ROWS"
    motifs = {str(m) for row in rows for m in row.get("failure_motif_tags", [])}
    if len(rows) >= 5 and len(motifs) >= 3 and bool(family_weights):
        return "DEEPENING_MULTI_MOTIF"
    if len(rows) >= 1 and bool(family_weights):
        return "EARLY_BUT_STRUCTURED"
    return "PARTIAL_MEMORY_ONLY"

