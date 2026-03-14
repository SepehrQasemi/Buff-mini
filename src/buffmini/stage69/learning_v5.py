"""Stage-69 campaign memory v5."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_campaign_memory_rows_v5(
    *,
    outcomes: pd.DataFrame,
    gated_candidates: pd.DataFrame,
    stage28_run_id: str,
) -> list[dict[str, Any]]:
    frame = outcomes.copy() if isinstance(outcomes, pd.DataFrame) else pd.DataFrame()
    gated_ids = set(gated_candidates.get("candidate_id", pd.Series(dtype=str)).astype(str).tolist()) if isinstance(gated_candidates, pd.DataFrame) else set()
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for rec in frame.to_dict(orient="records"):
        cid = str(rec.get("candidate_id", ""))
        cost_edge = float(rec.get("cost_edge_proxy", 0.0) or 0.0)
        tp_label = float(rec.get("tp_before_sl_label", 0.0) or 0.0)
        net_label = float(rec.get("expected_net_after_cost_label", 0.0) or 0.0)
        if tp_label <= 0.0:
            guidance = "reshape_trigger_composition"
        elif net_label <= 0.0:
            guidance = "prioritize_high_edge_per_trade_setups"
        elif cid not in gated_ids:
            guidance = "strengthen_confirmation_logic"
        else:
            guidance = "alter_geometry_and_invalidation"
        rows.append(
            {
                "run_id": str(stage28_run_id),
                "candidate_id": cid,
                "family": str(rec.get("family", "unknown")),
                "timeframe": str(rec.get("timeframe", "unknown")),
                "cost_drag_score": float(max(0.0, -cost_edge)),
                "reuse_score": float(1.0 if cid in gated_ids else 0.0),
                "mutation_guidance": guidance,
                "allocation_only": True,
            }
        )
    return rows


def derive_campaign_priors_v5(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"family_allocation": {}, "timeframe_allocation": {}, "threshold_prior": 0.55}
    family_alloc = {
        str(name): float(round(grp["reuse_score"].mean() + 0.5 - grp["cost_drag_score"].mean(), 8))
        for name, grp in frame.groupby("family", dropna=False)
    }
    timeframe_alloc = {
        str(name): float(round(grp["reuse_score"].mean() + 0.5 - grp["cost_drag_score"].mean(), 8))
        for name, grp in frame.groupby("timeframe", dropna=False)
    }
    threshold_prior = float(round(min(0.8, max(0.5, 0.55 + frame["cost_drag_score"].mean() * 0.1)), 8))
    return {
        "family_allocation": family_alloc,
        "timeframe_allocation": timeframe_alloc,
        "threshold_prior": threshold_prior,
    }

