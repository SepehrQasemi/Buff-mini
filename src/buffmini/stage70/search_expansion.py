"""Stage-70 deterministic search space expansion with structured mechanisms."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buffmini.research.mechanisms import generate_mechanism_source_candidates, mechanism_families
from buffmini.utils.hashing import stable_hash


EXPANDED_FAMILIES: tuple[str, ...] = mechanism_families()


def economic_fingerprint(record: dict[str, Any]) -> str:
    material = {
        "family": str(record.get("family", "")),
        "subfamily": str(record.get("subfamily", "")),
        "timeframe": str(record.get("timeframe", "")),
        "context": str(record.get("context", "")),
        "trigger": str(record.get("trigger", "")),
        "confirmation": str(record.get("confirmation", "")),
        "participation_style": str(record.get("participation_style", record.get("participation", ""))),
        "invalidation": str(record.get("invalidation", "")),
        "risk_model": str(record.get("risk_model", "")),
        "exit_family": str(record.get("exit_family", "")),
        "time_stop_bars": int(record.get("time_stop_bars", 0)),
        "session_filter": str(record.get("session_filter", "")),
        "expected_regime": str(record.get("expected_regime", "")),
    }
    return str(stable_hash(material, length=20))


def similarity_bucket(record: dict[str, Any]) -> str:
    material = {
        "family": str(record.get("family", "")),
        "subfamily": str(record.get("subfamily", "")),
        "timeframe": str(record.get("timeframe", "")),
        "context": str(record.get("context", "")),
        "trigger": str(record.get("trigger", "")),
        "expected_regime": str(record.get("expected_regime", "")),
        "risk_model": str(record.get("risk_model", "")),
        "exit_family": str(record.get("exit_family", "")),
    }
    return str(stable_hash(material, length=18))


def deduplicate_economic_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return frame
    if "economic_fingerprint" not in frame.columns:
        frame["economic_fingerprint"] = [economic_fingerprint(dict(row)) for row in frame.to_dict(orient="records")]
    frame["priority_seed"] = pd.to_numeric(frame.get("priority_seed", 0.0), errors="coerce").fillna(0.0)
    frame["novelty_score"] = pd.to_numeric(frame.get("novelty_score", 0.0), errors="coerce").fillna(0.0)
    ranked = frame.sort_values(["priority_seed", "novelty_score", "candidate_id"], ascending=[False, False, True]).drop_duplicates(
        subset=["economic_fingerprint"],
        keep="first",
    )
    return ranked.reset_index(drop=True)


def collapse_similarity_candidates(candidates: pd.DataFrame, *, max_per_bucket: int = 3) -> pd.DataFrame:
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return frame
    frame["similarity_bucket"] = [similarity_bucket(dict(row)) for row in frame.to_dict(orient="records")]
    frame["priority_seed"] = pd.to_numeric(frame.get("priority_seed", 0.0), errors="coerce").fillna(0.0)
    frame["novelty_score"] = pd.to_numeric(frame.get("novelty_score", 0.0), errors="coerce").fillna(0.0)
    frame = frame.sort_values(["priority_seed", "novelty_score", "candidate_id"], ascending=[False, False, True]).reset_index(drop=True)
    frame["_bucket_rank"] = frame.groupby("similarity_bucket", dropna=False).cumcount()
    frame["similarity_penalty"] = frame["_bucket_rank"].astype(float) / max(1.0, float(max_per_bucket))
    frame["duplication_score"] = frame["similarity_penalty"].clip(lower=0.0, upper=1.0)
    collapsed = frame.loc[frame["_bucket_rank"] < int(max_per_bucket), :].drop(columns=["_bucket_rank"])
    return collapsed.reset_index(drop=True)


def generate_expanded_candidates(
    *,
    discovery_timeframes: list[str],
    budget_mode_selected: str,
    active_families: list[str] | None = None,
    failure_feedback: dict[str, Any] | None = None,
    min_search_candidates: int = 2500,
    min_full_audit_candidates: int = 10000,
) -> pd.DataFrame:
    mode = str(budget_mode_selected).lower().strip()
    target = int(min_full_audit_candidates if mode == "full_audit" else min_search_candidates)
    timeframes = [str(v) for v in discovery_timeframes if str(v).strip()] or ["1h"]
    raw = generate_mechanism_source_candidates(
        discovery_timeframes=timeframes,
        budget_mode_selected=mode,
        active_families=active_families,
        target_min_candidates=int(target),
    )
    raw = raw.copy()
    raw["hypothesis"] = [
        {
            "context": row.get("context"),
            "trigger": row.get("trigger"),
            "participation": row.get("participation_style", row.get("participation")),
            "invalidation": row.get("invalidation"),
            "risk_model": row.get("risk_model"),
            "exit_family": row.get("exit_family"),
            "time_stop_bars": int(row.get("time_stop_bars", 0)),
            "session_filter": row.get("session_filter"),
        }
        for row in raw.to_dict(orient="records")
    ]
    raw["economic_fingerprint"] = [economic_fingerprint(dict(row)) for row in raw.to_dict(orient="records")]
    seeded = _apply_failure_feedback(raw, failure_feedback or {})
    max_per_bucket = 4 if mode == "full_audit" else 3
    collapsed = collapse_similarity_candidates(seeded, max_per_bucket=max_per_bucket)
    deduped = deduplicate_economic_candidates(collapsed)
    while len(deduped) < int(target) and max_per_bucket < 8:
        max_per_bucket += 1
        collapsed = collapse_similarity_candidates(seeded, max_per_bucket=max_per_bucket)
        deduped = deduplicate_economic_candidates(collapsed)
    out = deduped.sort_values(["priority_seed", "novelty_score", "candidate_id"], ascending=[False, False, True]).head(target).reset_index(drop=True)
    return out


def _apply_failure_feedback(frame: pd.DataFrame, feedback: dict[str, Any]) -> pd.DataFrame:
    work = frame.copy()
    family_adjustments = {str(key): float(value) for key, value in dict(feedback.get("family_priority_adjustments", {})).items()}
    guidance = [str(item) for item in list(feedback.get("threshold_guidance", []))]
    if not work.empty:
        work["priority_seed"] = pd.to_numeric(work.get("priority_seed", 0.0), errors="coerce").fillna(0.0)
        work["novelty_score"] = pd.to_numeric(work.get("novelty_score", 0.0), errors="coerce").fillna(0.0)
        work["transfer_risk_prior"] = pd.to_numeric(work.get("transfer_risk_prior", 0.0), errors="coerce").fillna(0.0)
        work["priority_seed"] = work["priority_seed"] + work.get("family", "").astype(str).map(lambda family: family_adjustments.get(str(family), 0.0)).fillna(0.0)
        if "favor_lower_transfer_risk" in guidance:
            work["priority_seed"] = work["priority_seed"] + (0.04 * (1.0 - work["transfer_risk_prior"].clip(lower=0.0, upper=1.0)))
        if "diversify_mechanism_mix" in guidance:
            work["novelty_score"] = work["novelty_score"] + 0.03
        if "raise_cost_edge_floor" in guidance:
            work["priority_seed"] = work["priority_seed"] + work.get("risk_model", "").astype(str).map(lambda _: 0.01).fillna(0.0)
        if "broaden_trade_density" in guidance:
            work["novelty_score"] = work["novelty_score"] + work.get("session_filter", "").astype(str).map(
                lambda value: 0.02 if value == "any_session" else 0.0
            ).fillna(0.0)
        work["priority_seed"] = work["priority_seed"].clip(lower=0.0, upper=2.0)
        work["novelty_score"] = work["novelty_score"].clip(lower=0.0, upper=2.0)
    return work
