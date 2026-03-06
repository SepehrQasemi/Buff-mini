"""Stage-42 failure-aware self-learning upgrades."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buffmini.stage37.self_learning import compute_family_exploration_weights


def expand_registry_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    raw_candidate_count: int,
    shortlisted_count: int,
    mutation_origin: str = "stage42_upgrade",
) -> list[dict[str, Any]]:
    """Expand Stage-37 rows to Stage-42 memory schema."""

    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["seed"] = int(seed)
        item["context"] = str(item.get("context", "global"))
        item["raw_candidate_count"] = int(item.get("raw_signal_count", raw_candidate_count))
        item["shortlisted_count"] = int(shortlisted_count)
        item["mutation_origin"] = str(item.get("mutation_origin", mutation_origin))
        item["mutation_guidance"] = failure_aware_mutation_action(item)
        motifs = list(item.get("failure_motif_tags", []))
        if not motifs:
            motifs = [f"REJECT::{str(item.get('top_reject_reason', 'unknown')).upper()}"]
        item["failure_motif_tags"] = [str(v) for v in motifs]
        out.append(item)
    return out


def failure_aware_mutation_action(row: dict[str, Any]) -> str:
    """Choose mutation strategy based on observed failure pattern."""

    raw = int(row.get("raw_candidate_count", row.get("raw_signal_count", 0)))
    activation = float(row.get("activation_rate", 0.0))
    final_trades = int(row.get("final_trade_count", 0))
    cost_fail = float(row.get("cost_gate_fail_rate", 0.0))
    feas_fail = float(row.get("feasibility_fail_rate", 0.0))
    exp_lcb = float(row.get("exp_lcb", 0.0))
    status = str(row.get("status", "")).strip().lower()

    if raw <= 0:
        return "widen_context_and_expand_grammar"
    if activation > 0.0 and cost_fail >= 0.40:
        return "mutate_threshold_exit_and_cost_sensitivity"
    if activation > 0.0 and final_trades <= 0:
        return "mutate_composer_and_policy_layer"
    if activation > 0.0 and exp_lcb <= 0.0:
        return "broaden_windows_and_context_definition"
    if status == "dead_end" or feas_fail >= 0.60:
        return "downweight_family_and_reduce_exploration"
    return "explore_local_variants_around_elites"


def family_memory_summary(rows: list[dict[str, Any]], *, top_k: int = 5) -> dict[str, Any]:
    """Build top-config/dead-branch/motif memory with exploration weights."""

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "family_weights": {},
            "top_configs_per_family": {},
            "dead_branches_per_family": {},
            "recurring_failure_motifs": {},
        }
    frame["family"] = frame.get("family", "").astype(str)
    frame["exp_lcb"] = pd.to_numeric(frame.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
    frame["activation_rate"] = pd.to_numeric(frame.get("activation_rate", 0.0), errors="coerce").fillna(0.0)
    frame["final_trade_count"] = pd.to_numeric(frame.get("final_trade_count", 0), errors="coerce").fillna(0).astype(int)
    if "status" not in frame.columns:
        frame["status"] = "active"
    else:
        frame["status"] = frame["status"].astype(str)

    top_configs: dict[str, list[dict[str, Any]]] = {}
    dead_branches: dict[str, int] = {}
    motifs: dict[str, int] = {}

    for family, grp in frame.groupby("family", dropna=False):
        ordered = grp.sort_values(
            ["exp_lcb", "activation_rate", "final_trade_count", "feature_subset_signature"],
            ascending=[False, False, False, True],
        ).head(int(max(1, top_k)))
        top_configs[str(family)] = ordered.to_dict(orient="records")
        dead_branches[str(family)] = int((grp["status"].str.lower() == "dead_end").sum())

    for row in rows:
        tags = list(row.get("failure_motif_tags", []))
        if not tags:
            tags = [f"REJECT::{str(row.get('top_reject_reason', 'unknown')).upper()}"]
        for tag in tags:
            key = str(tag)
            motifs[key] = int(motifs.get(key, 0) + 1)

    weights = compute_family_exploration_weights(rows)
    return {
        "family_weights": weights,
        "top_configs_per_family": top_configs,
        "dead_branches_per_family": dead_branches,
        "recurring_failure_motifs": dict(sorted(motifs.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))),
    }


def stability_aware_feature_pruning(
    history_rows: list[dict[str, Any]],
    *,
    min_runs: int = 2,
    min_mean_contribution: float = 0.0,
) -> dict[str, Any]:
    """Prune features only when repeatedly weak across runs."""

    frame = pd.DataFrame(history_rows)
    if frame.empty:
        return {"kept_features": [], "dropped_features": [], "contribution_stats": {}}
    frame["feature"] = frame.get("feature", "").astype(str)
    frame["run_id"] = frame.get("run_id", "").astype(str)
    frame["gain"] = pd.to_numeric(frame.get("gain", 0.0), errors="coerce").fillna(0.0)
    grouped = frame.groupby("feature", dropna=False)["gain"].agg(["mean", "count"]).reset_index()

    keep: list[str] = []
    drop: list[str] = []
    for row in grouped.to_dict(orient="records"):
        feature = str(row["feature"])
        count = int(row["count"])
        mean_gain = float(row["mean"])
        if count >= int(min_runs) and mean_gain <= float(min_mean_contribution):
            drop.append(feature)
        else:
            keep.append(feature)

    stats = {
        str(row["feature"]): {"mean_gain": float(row["mean"]), "observations": int(row["count"])}
        for row in grouped.to_dict(orient="records")
    }
    return {
        "kept_features": sorted(keep),
        "dropped_features": sorted(drop),
        "contribution_stats": stats,
    }


def build_self_diagnosis(*, previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    """Build machine-readable diagnosis of improvements/regressions/next moves."""

    prev = previous or {}
    curr_stage_b = int(((current.get("counts", {}) or {}).get("stage_b", 0)))
    prev_stage_b = int(((prev.get("counts", {}) or {}).get("stage_b", 0)))
    curr_raw = int(current.get("raw_candidate_count", 0))
    prev_raw = int(prev.get("raw_candidate_count", 0))

    improved: list[str] = []
    regressed: list[str] = []
    if curr_raw > prev_raw:
        improved.append("raw_candidate_generation")
    elif curr_raw < prev_raw:
        regressed.append("raw_candidate_generation")
    if curr_stage_b > prev_stage_b:
        improved.append("stage_b_survival")
    elif curr_stage_b < prev_stage_b:
        regressed.append("stage_b_survival")

    family_weights = dict((current.get("family_memory", {}) or {}).get("family_weights", {}))
    explore_less = sorted([k for k, v in family_weights.items() if float(v) <= 0.25])
    explore_more = sorted([k for k, v in family_weights.items() if float(v) >= 1.0])
    return {
        "improved": improved,
        "regressed": regressed,
        "mutate_next": str(current.get("global_mutation_guidance", "widen_context_and_expand_grammar")),
        "explore_less": explore_less,
        "explore_more": explore_more,
    }
