"""Stage-47 setup-based candidate generation with deterministic beam search."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buffmini.utils.hashing import stable_hash


SETUP_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "family": "structure_pullback_continuation",
        "contexts": ("trend", "flow-dominant"),
        "trigger": "pullback_to_structure_level",
        "confirmation": "flow_continuation",
        "invalidation": "structure_break",
        "modules": ("structure_engine", "flow_regime_engine"),
    },
    {
        "family": "liquidity_sweep_reversal",
        "contexts": ("range", "exhaustion", "sentiment-extreme"),
        "trigger": "liquidity_sweep",
        "confirmation": "reversal_impulse",
        "invalidation": "failed_reclaim",
        "modules": ("liquidity_map", "trade_geometry_layer"),
    },
    {
        "family": "squeeze_flow_breakout",
        "contexts": ("squeeze", "shock"),
        "trigger": "squeeze_release",
        "confirmation": "flow_burst_alignment",
        "invalidation": "failed_expansion",
        "modules": ("volatility_regime_engine", "flow_regime_engine"),
    },
    {
        "family": "crowded_side_squeeze",
        "contexts": ("sentiment-extreme", "funding-stress", "shock"),
        "trigger": "crowding_extreme",
        "confirmation": "volatility_release",
        "invalidation": "crowding_reinforcement",
        "modules": ("crowding_layer", "volatility_regime_engine"),
    },
    {
        "family": "flow_exhaustion_reversal",
        "contexts": ("exhaustion", "shock"),
        "trigger": "flow_exhaustion",
        "confirmation": "structure_reclaim",
        "invalidation": "fresh_flow_burst",
        "modules": ("flow_regime_engine", "structure_engine"),
    },
    {
        "family": "regime_shift_entry",
        "contexts": ("shock", "trend", "range"),
        "trigger": "context_shift",
        "confirmation": "mtf_alignment",
        "invalidation": "alignment_break",
        "modules": ("mtf_bias_completion", "structure_engine"),
    },
)


def validate_setup_candidate(candidate: dict[str, Any]) -> None:
    required = {
        "candidate_id",
        "family",
        "context",
        "trigger",
        "confirmation",
        "invalidation",
        "source_candidate_id",
        "beam_score",
        "modules",
        "lineage",
    }
    missing = sorted(required.difference(set(candidate.keys())))
    if missing:
        raise ValueError(f"Missing setup candidate keys: {missing}")
    for key in ("family", "context", "trigger", "confirmation", "invalidation", "source_candidate_id"):
        if not str(candidate.get(key, "")).strip():
            raise ValueError(f"{key} must be non-empty")
    if not isinstance(candidate.get("modules"), list):
        raise ValueError("modules must be list")
    lineage = candidate.get("lineage", {})
    if not isinstance(lineage, dict):
        raise ValueError("lineage must be object")
    for key in ("source_candidate_id", "source_branch", "modules"):
        if key not in lineage:
            raise ValueError(f"lineage.{key} missing")


def generate_setup_candidates(
    layer_a: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic setup candidates from Stage-39 Layer-A rows."""

    if not isinstance(layer_a, pd.DataFrame) or layer_a.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "family",
                "context",
                "trigger",
                "confirmation",
                "invalidation",
                "source_candidate_id",
                "source_branch",
                "beam_score",
                "modules",
                "lineage",
            ]
        )
    rows: list[dict[str, Any]] = []
    for src in layer_a.to_dict(orient="records"):
        base_context = str(src.get("broad_context", "range"))
        source_id = str(src.get("candidate_id", ""))
        source_branch = str(src.get("branch", "unknown"))
        src_score = float(src.get("layer_score", 0.0))
        for idx, family in enumerate(SETUP_FAMILIES):
            if base_context not in set(str(v) for v in family["contexts"]):
                continue
            diversity_bonus = 0.02 * float((idx % 3) + 1)
            score = float(round(src_score * 0.7 + diversity_bonus + 0.1, 6))
            cid = stable_hash(
                {
                    "seed": int(seed),
                    "source_candidate_id": source_id,
                    "setup_family": str(family["family"]),
                    "context": base_context,
                    "source_branch": source_branch,
                },
                length=16,
            )
            row = {
                "candidate_id": f"s47_{cid}",
                "family": str(family["family"]),
                "context": base_context,
                "trigger": str(family["trigger"]),
                "confirmation": str(family["confirmation"]),
                "invalidation": str(family["invalidation"]),
                "source_candidate_id": source_id,
                "source_branch": source_branch,
                "beam_score": score,
                "modules": [str(v) for v in family["modules"]],
                "lineage": {
                    "source_candidate_id": source_id,
                    "source_branch": source_branch,
                    "modules": [str(v) for v in family["modules"]],
                    "context": base_context,
                },
            }
            validate_setup_candidate(row)
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["beam_score", "family", "candidate_id"], ascending=[False, True, True]).reset_index(drop=True)


def beam_search_setups(
    setups: pd.DataFrame,
    *,
    beam_width: int = 64,
    per_family_max: int = 12,
) -> pd.DataFrame:
    """Deterministic beam search with diversity-aware per-family caps."""

    if not isinstance(setups, pd.DataFrame) or setups.empty:
        return pd.DataFrame(columns=list(setups.columns) if isinstance(setups, pd.DataFrame) else [])
    ranked = setups.copy()
    ranked["beam_score"] = pd.to_numeric(ranked.get("beam_score", 0.0), errors="coerce").fillna(0.0)
    ranked = ranked.sort_values(["beam_score", "family", "candidate_id"], ascending=[False, True, True]).reset_index(drop=True)
    ranked["family_rank"] = ranked.groupby("family", dropna=False).cumcount()
    pruned = ranked.loc[ranked["family_rank"] < int(max(1, per_family_max)), :].drop(columns=["family_rank"])
    return pruned.head(int(max(1, beam_width))).reset_index(drop=True)


def summarize_stage47_candidates(
    *,
    baseline_raw_candidate_count: int,
    setups: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> dict[str, Any]:
    """Build Stage-47 summary payload core fields."""

    setup_family_counts = (
        {str(k): int(v) for k, v in shortlist["family"].astype(str).value_counts(dropna=False).to_dict().items()}
        if not shortlist.empty and "family" in shortlist.columns
        else {}
    )
    context_counts = (
        {str(k): int(v) for k, v in shortlist["context"].astype(str).value_counts(dropna=False).to_dict().items()}
        if not shortlist.empty and "context" in shortlist.columns
        else {}
    )
    lineage_examples_present = bool(not shortlist.empty and "lineage" in shortlist.columns and shortlist["lineage"].notna().any())
    return {
        "baseline_raw_candidate_count": int(baseline_raw_candidate_count),
        "upgraded_raw_candidate_count": int(setups.shape[0]) if isinstance(setups, pd.DataFrame) else 0,
        "shortlisted_count": int(shortlist.shape[0]) if isinstance(shortlist, pd.DataFrame) else 0,
        "setup_family_counts": setup_family_counts,
        "context_counts": context_counts,
        "lineage_examples_present": lineage_examples_present,
    }

