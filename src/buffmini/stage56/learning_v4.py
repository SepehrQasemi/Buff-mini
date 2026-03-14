"""Stage-56 self-learning v4 memory and allocation logic."""

from __future__ import annotations

from typing import Any

import pandas as pd


_REQUIRED_ROW_FIELDS: tuple[str, ...] = (
    "run_id",
    "seed",
    "family",
    "timeframe",
    "context",
    "trigger_composition",
    "geometry_signature",
    "cost_drag_score",
    "stage_a_failures",
    "stage_b_failures",
    "replay_outcome",
    "wf_outcome",
    "mc_outcome",
    "reuse_score",
    "mutation_guidance",
)


def validate_registry_row_v4(row: dict[str, Any]) -> None:
    missing = [field for field in _REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Missing learning_registry_v4 fields: {missing}")
    _ = float(row.get("cost_drag_score", 0.0))
    _ = float(row.get("reuse_score", 0.0))
    if not isinstance(row.get("stage_a_failures"), list):
        raise ValueError("stage_a_failures must be list")
    if not isinstance(row.get("stage_b_failures"), list):
        raise ValueError("stage_b_failures must be list")


def expand_registry_rows_v4(
    *,
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    seed: int,
    run_id: str,
) -> list[dict[str, Any]]:
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    preds = predictions.copy() if isinstance(predictions, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return []
    merged = frame.merge(preds, on="candidate_id", how="left")
    rows: list[dict[str, Any]] = []
    for record in merged.to_dict(orient="records"):
        stage_a_failures = []
        stage_b_failures = []
        pre_reject_raw = record.get("pre_replay_reject_reason", "")
        if pd.isna(pre_reject_raw) if not isinstance(pre_reject_raw, (list, dict)) else False:
            pre_reject = ""
        else:
            pre_reject = str(pre_reject_raw).strip()
        if pre_reject:
            stage_a_failures.append(pre_reject)
        if float(record.get("expected_net_after_cost", 0.0) or 0.0) <= 0.0:
            stage_a_failures.append("REJECT::COST_MARGIN_TOO_LOW")
        if float(record.get("exp_lcb_proxy", 0.0) or 0.0) <= 0.0:
            stage_b_failures.append("REJECT::NO_SIGNAL")
        row = {
            "run_id": str(run_id),
            "seed": int(seed),
            "family": str(record.get("family", "__unknown__")),
            "timeframe": str(record.get("timeframe", "__unknown__")),
            "context": record.get("context", {}),
            "trigger_composition": record.get("trigger", {}),
            "geometry_signature": {
                "entry_logic": str(record.get("entry_logic", "")),
                "stop_logic": str(record.get("stop_logic", "")),
                "target_logic": str(record.get("target_logic", "")),
                "first_target_rr": float((record.get("rr_model") or {}).get("first_target_rr", 0.0)) if isinstance(record.get("rr_model"), dict) else 0.0,
            },
            "cost_drag_score": float(round(max(0.0, -float(record.get("expected_net_after_cost", 0.0) or 0.0)), 8)),
            "stage_a_failures": stage_a_failures,
            "stage_b_failures": stage_b_failures,
            "replay_outcome": "PASS" if float(record.get("replay_priority", 0.0) or 0.0) >= float(preds.get("replay_priority", pd.Series(dtype=float)).median() if not preds.empty else 0.0) else "FAIL",
            "wf_outcome": "PASS" if float(record.get("exp_lcb_proxy", 0.0) or 0.0) > 0.0 else "FAIL",
            "mc_outcome": "PASS" if float(record.get("mfe_pct", 0.0) or 0.0) > abs(float(record.get("mae_pct", 0.0) or 0.0)) else "FAIL",
            "reuse_score": float(round(min(1.0, max(0.0, float(record.get("replay_priority", 0.0) or 0.0))), 8)),
            "mutation_guidance": "",
        }
        row["mutation_guidance"] = mutation_guidance_v4(row)
        validate_registry_row_v4(row)
        rows.append(row)
    return rows


def mutation_guidance_v4(row: dict[str, Any]) -> str:
    motifs = [str(v) for v in row.get("stage_a_failures", []) + row.get("stage_b_failures", [])]
    if any(motif in {"REJECT::WEAK_TRIGGER", "REJECT::NO_SIGNAL"} for motif in motifs):
        return "reshape_trigger_composition"
    if any(motif in {"REJECT::BAD_GEOMETRY", "REJECT::BAD_RR"} for motif in motifs):
        return "alter_geometry_and_invalidation"
    if any(motif in {"REJECT::COST_MARGIN_TOO_LOW", "REJECT::COST_DRAG"} for motif in motifs):
        return "prioritize_high_edge_per_trade_setups"
    if any(motif in {"REJECT::NO_CONFIRMATION", "REJECT::FAILED_LIQUIDITY_CONFIRMATION", "REJECT::WEAK_FLOW_CONTEXT"} for motif in motifs):
        return "strengthen_confirmation_logic"
    return "widen_context_and_expand_grammar"


def derive_allocation_adjustments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "family_allocation": {},
            "timeframe_allocation": {},
            "threshold_priors": {},
            "replay_budget_allocation": {},
        }
    frame["reuse_score"] = pd.to_numeric(frame.get("reuse_score", 0.0), errors="coerce").fillna(0.0)
    frame["cost_drag_score"] = pd.to_numeric(frame.get("cost_drag_score", 0.0), errors="coerce").fillna(0.0)
    family_allocation = {
        str(name): float(round(max(0.1, grp["reuse_score"].mean() + 0.5 - grp["cost_drag_score"].mean()), 8))
        for name, grp in frame.groupby("family", dropna=False)
    }
    timeframe_allocation = {
        str(name): float(round(max(0.1, grp["reuse_score"].mean() + 0.5 - grp["cost_drag_score"].mean()), 8))
        for name, grp in frame.groupby("timeframe", dropna=False)
    }
    threshold_priors = {
        "stage_a_prob_threshold": float(round(min(0.75, max(0.50, 0.55 + frame["cost_drag_score"].mean() * 0.1)), 8)),
        "min_rr": float(round(max(1.5, 1.5 + frame["cost_drag_score"].mean() * 2.0), 8)),
    }
    replay_budget_allocation = {
        str(row["family"]): float(round(float(row["reuse_score"]) / max(frame["reuse_score"].sum(), 1e-9), 8))
        for row in frame.sort_values(["reuse_score", "family"], ascending=[False, True]).head(min(5, len(frame))).to_dict(orient="records")
    }
    return {
        "family_allocation": family_allocation,
        "timeframe_allocation": timeframe_allocation,
        "threshold_priors": threshold_priors,
        "replay_budget_allocation": replay_budget_allocation,
    }


def assess_learning_depth_v4(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "SHALLOW_NO_ROWS"
    motifs = {str(motif) for row in rows for motif in row.get("stage_a_failures", []) + row.get("stage_b_failures", [])}
    if len(rows) >= 5 and len(motifs) >= 3:
        return "DEEPENING_MULTI_SIGNAL_MEMORY"
    if len(rows) >= 1:
        return "EARLY_BUT_STRUCTURED"
    return "PARTIAL"
