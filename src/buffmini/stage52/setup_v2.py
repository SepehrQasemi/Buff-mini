"""Stage-52 setup candidate v2 contract and geometry generation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buffmini.stage51 import DEFAULT_DISCOVERY_TIMEFRAMES, DEFAULT_SETUP_FAMILIES
from buffmini.utils.hashing import stable_hash


ALLOWED_SETUP_FAMILIES: tuple[str, ...] = tuple(DEFAULT_SETUP_FAMILIES)
ALLOWED_DISCOVERY_TIMEFRAMES: tuple[str, ...] = tuple(DEFAULT_DISCOVERY_TIMEFRAMES)

_REQUIRED_KEYS: tuple[str, ...] = (
    "candidate_id",
    "family",
    "timeframe",
    "context",
    "trigger",
    "confirmation",
    "invalidation",
    "entry_logic",
    "stop_logic",
    "target_logic",
    "hold_logic",
    "rr_model",
    "cost_edge_proxy",
    "geometry",
    "lineage",
    "pre_replay_reject_reason",
    "eligible_for_replay",
    "economic_fingerprint",
)

_FAMILY_GEOMETRY: dict[str, dict[str, Any]] = {
    "structure_pullback_continuation": {
        "entry_logic": "enter_on_pullback_reclaim",
        "stop_logic": "below_structure_swing",
        "target_logic": "trend_leg_extension",
        "hold_logic": "hold_until_target_or_structure_break",
        "stop_pct": 0.0060,
        "first_target_pct": 0.0105,
        "stretch_target_pct": 0.0170,
        "hold_bars": 12,
    },
    "liquidity_sweep_reversal": {
        "entry_logic": "enter_on_reclaim_after_sweep",
        "stop_logic": "beyond_sweep_extreme",
        "target_logic": "mean_reversion_then_extension",
        "hold_logic": "hold_until_reclaim_fails_or_target_hits",
        "stop_pct": 0.0055,
        "first_target_pct": 0.0115,
        "stretch_target_pct": 0.0185,
        "hold_bars": 8,
    },
    "squeeze_flow_breakout": {
        "entry_logic": "enter_on_breakout_close_with_flow",
        "stop_logic": "inside_squeeze_reentry",
        "target_logic": "volatility_expansion_projection",
        "hold_logic": "hold_while_flow_and_range_expansion_hold",
        "stop_pct": 0.0075,
        "first_target_pct": 0.0135,
        "stretch_target_pct": 0.0220,
        "hold_bars": 16,
    },
}

_TIMEFRAME_SCALE: dict[str, float] = {
    "15m": 0.75,
    "30m": 0.85,
    "1h": 1.00,
    "2h": 1.15,
    "4h": 1.35,
}


def validate_setup_candidate_v2(candidate: dict[str, Any]) -> None:
    missing = [key for key in _REQUIRED_KEYS if key not in candidate]
    if missing:
        raise ValueError(f"Missing Stage-52 keys: {missing}")
    for key in ("candidate_id", "family", "timeframe", "invalidation", "entry_logic", "stop_logic", "target_logic", "hold_logic"):
        if not str(candidate.get(key, "")).strip():
            raise ValueError(f"{key} must be non-empty")
    if str(candidate.get("family")) not in set(ALLOWED_SETUP_FAMILIES):
        raise ValueError("family must be Stage-52 active family")
    if str(candidate.get("timeframe")) not in set(ALLOWED_DISCOVERY_TIMEFRAMES):
        raise ValueError("timeframe must be Stage-52 discovery timeframe")
    if not isinstance(candidate.get("rr_model"), dict):
        raise ValueError("rr_model must be object")
    if not isinstance(candidate.get("geometry"), dict):
        raise ValueError("geometry must be object")
    if not isinstance(candidate.get("lineage"), dict):
        raise ValueError("lineage must be object")
    _ = float(candidate.get("cost_edge_proxy", 0.0))
    if not isinstance(candidate.get("eligible_for_replay"), bool):
        raise ValueError("eligible_for_replay must be bool")
    if not str(candidate.get("economic_fingerprint", "")).strip():
        raise ValueError("economic_fingerprint must be non-empty")


def build_setup_candidate_v2(
    source_candidate: dict[str, Any],
    *,
    timeframe: str,
    round_trip_cost_pct: float = 0.001,
    entry_reference_price: float = 1.0,
) -> dict[str, Any]:
    family = str(source_candidate.get("family", "")).strip()
    context = str(source_candidate.get("context", "unknown")).strip()
    trigger = str(source_candidate.get("trigger", "")).strip()
    confirmation = str(source_candidate.get("confirmation", "")).strip()
    invalidation = str(source_candidate.get("invalidation", "")).strip()
    source_candidate_id = str(source_candidate.get("candidate_id", source_candidate.get("source_candidate_id", ""))).strip()
    beam_score = float(pd.to_numeric(pd.Series([source_candidate.get("beam_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    geometry = _build_geometry(
        family=family,
        timeframe=str(timeframe),
        entry_reference_price=float(entry_reference_price),
        round_trip_cost_pct=float(round_trip_cost_pct),
    )
    rr_model = {
        "first_target_rr": float(geometry["first_target_pct"] / max(geometry["stop_distance_pct"], 1e-9)),
        "stretch_target_rr": float(geometry["stretch_target_pct"] / max(geometry["stop_distance_pct"], 1e-9)),
    }
    cost_edge_proxy = float(geometry["first_target_pct"] - (float(round_trip_cost_pct) * 1.5))
    reject_reason = _pre_replay_reject_reason(
        family=family,
        timeframe=str(timeframe),
        trigger=trigger,
        confirmation=confirmation,
        invalidation=invalidation,
        rr_model=rr_model,
        cost_edge_proxy=cost_edge_proxy,
    )
    payload = {
        "candidate_id": f"s52_{stable_hash({'source_candidate_id': source_candidate_id, 'family': family, 'timeframe': timeframe}, length=16)}",
        "family": family,
        "timeframe": str(timeframe),
        "context": {
            "primary": context,
            "beam_score": float(round(max(0.0, beam_score), 6)),
            "modules": [str(v) for v in source_candidate.get("modules", [])] if isinstance(source_candidate.get("modules"), list) else [],
        },
        "trigger": {"type": trigger, "strength": float(round(max(0.0, min(1.0, beam_score)), 6))},
        "confirmation": {"type": confirmation, "required": True},
        "invalidation": invalidation,
        "entry_logic": str(geometry["entry_logic"]),
        "stop_logic": str(geometry["stop_logic"]),
        "target_logic": str(geometry["target_logic"]),
        "hold_logic": str(geometry["hold_logic"]),
        "rr_model": rr_model,
        "cost_edge_proxy": float(round(cost_edge_proxy, 8)),
        "geometry": geometry,
        "lineage": {
            "source_candidate_id": source_candidate_id,
            "source_branch": str(source_candidate.get("source_branch", source_candidate.get("branch", "unknown"))),
            "stage": "52",
            "family": family,
        },
        "source_candidate_id": source_candidate_id,
        "beam_score": float(round(beam_score, 6)),
        "pre_replay_reject_reason": str(reject_reason),
        "eligible_for_replay": bool(not reject_reason),
    }
    payload["economic_fingerprint"] = build_economic_fingerprint(payload)
    validate_setup_candidate_v2(payload)
    return payload


def default_geometry_for_family(
    *,
    family: str,
    timeframe: str,
    round_trip_cost_pct: float = 0.001,
    entry_reference_price: float = 1.0,
) -> dict[str, Any]:
    """Return deterministic default geometry for one supported Stage-52 family/timeframe."""

    return _build_geometry(
        family=str(family),
        timeframe=str(timeframe),
        entry_reference_price=float(entry_reference_price),
        round_trip_cost_pct=float(round_trip_cost_pct),
    )


def upgrade_candidates_to_v2(
    candidates: pd.DataFrame,
    *,
    timeframe: str,
    round_trip_cost_pct: float = 0.001,
) -> pd.DataFrame:
    if not isinstance(candidates, pd.DataFrame) or candidates.empty:
        return pd.DataFrame(columns=list(_REQUIRED_KEYS))
    rows = [
        build_setup_candidate_v2(
            dict(row),
            timeframe=str(timeframe),
            round_trip_cost_pct=float(round_trip_cost_pct),
        )
        for row in candidates.to_dict(orient="records")
        if str(row.get("family", "")).strip() in set(ALLOWED_SETUP_FAMILIES)
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["eligible_for_replay", "beam_score", "candidate_id"], ascending=[False, False, True]).reset_index(drop=True)


def summarize_stage52_candidates(candidates: pd.DataFrame) -> dict[str, Any]:
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    rejection_counts = (
        {
            str(key): int(value)
            for key, value in frame.loc[frame["pre_replay_reject_reason"].astype(str) != "", "pre_replay_reject_reason"]
            .astype(str)
            .value_counts(dropna=False)
            .to_dict()
            .items()
        }
        if not frame.empty and "pre_replay_reject_reason" in frame.columns
        else {}
    )
    timeframe_counts = (
        {str(key): int(value) for key, value in frame["timeframe"].astype(str).value_counts(dropna=False).to_dict().items()}
        if not frame.empty and "timeframe" in frame.columns
        else {}
    )
    family_counts = (
        {str(key): int(value) for key, value in frame["family"].astype(str).value_counts(dropna=False).to_dict().items()}
        if not frame.empty and "family" in frame.columns
        else {}
    )
    avg_cost_edge = float(pd.to_numeric(frame.get("cost_edge_proxy", 0.0), errors="coerce").fillna(0.0).mean()) if not frame.empty else 0.0
    economic_fingerprint_count = int(frame.get("economic_fingerprint", pd.Series(dtype=str)).astype(str).nunique(dropna=True)) if not frame.empty else 0
    return {
        "status": "SUCCESS" if not frame.empty else "PARTIAL",
        "candidate_count": int(frame.shape[0]),
        "eligible_for_replay_count": int(frame.get("eligible_for_replay", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not frame.empty else 0,
        "economic_fingerprint_count": economic_fingerprint_count,
        "family_counts": family_counts,
        "timeframe_counts": timeframe_counts,
        "rejection_counts": rejection_counts,
        "avg_cost_edge_proxy": float(round(avg_cost_edge, 8)),
        "summary_hash": stable_hash(
            {
                "candidate_count": int(frame.shape[0]),
                "eligible_for_replay_count": int(frame.get("eligible_for_replay", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not frame.empty else 0,
                "economic_fingerprint_count": economic_fingerprint_count,
                "family_counts": family_counts,
                "timeframe_counts": timeframe_counts,
                "rejection_counts": rejection_counts,
                "avg_cost_edge_proxy": float(round(avg_cost_edge, 8)),
            },
            length=16,
        ),
    }


def build_economic_fingerprint(candidate: dict[str, Any]) -> str:
    rr_model = candidate.get("rr_model", {})
    geometry = candidate.get("geometry", {})
    material = {
        "family": str(candidate.get("family", "")),
        "timeframe": str(candidate.get("timeframe", "")),
        "entry_logic": str(candidate.get("entry_logic", "")),
        "stop_logic": str(candidate.get("stop_logic", "")),
        "target_logic": str(candidate.get("target_logic", "")),
        "hold_logic": str(candidate.get("hold_logic", "")),
        "first_target_rr": round(float((rr_model or {}).get("first_target_rr", 0.0)), 6),
        "expected_hold_bars": int((geometry or {}).get("expected_hold_bars", 0)),
        "cost_edge_bucket": round(float(candidate.get("cost_edge_proxy", 0.0)), 4),
    }
    return str(stable_hash(material, length=20))


def deduplicate_setup_candidates_by_economics(candidates: pd.DataFrame, *, keep_per_fingerprint: int = 1) -> pd.DataFrame:
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return frame
    if "economic_fingerprint" not in frame.columns:
        frame["economic_fingerprint"] = [build_economic_fingerprint(dict(row)) for row in frame.to_dict(orient="records")]
    frame["beam_score"] = pd.to_numeric(frame.get("beam_score", 0.0), errors="coerce").fillna(0.0)
    frame["cost_edge_proxy"] = pd.to_numeric(frame.get("cost_edge_proxy", 0.0), errors="coerce").fillna(0.0)
    frame["eligible_for_replay"] = frame.get("eligible_for_replay", False).fillna(False).astype(bool)
    ranked = frame.sort_values(
        ["eligible_for_replay", "beam_score", "cost_edge_proxy", "candidate_id"],
        ascending=[False, False, False, True],
    )
    keep = int(max(1, keep_per_fingerprint))
    out = ranked.groupby("economic_fingerprint", dropna=False, sort=False).head(keep)
    return out.reset_index(drop=True)


def evaluate_family_coverage(summary: dict[str, Any], *, active_families: list[str]) -> dict[str, Any]:
    family_counts = {
        str(key): int(value)
        for key, value in dict(summary.get("family_counts", {})).items()
    }
    missing_families = [
        str(family)
        for family in [str(item) for item in active_families]
        if int(family_counts.get(str(family), 0)) <= 0
    ]
    return {
        "family_coverage_ok": bool(not missing_families),
        "missing_families": missing_families,
    }


def _build_geometry(
    *,
    family: str,
    timeframe: str,
    entry_reference_price: float,
    round_trip_cost_pct: float,
) -> dict[str, Any]:
    if family not in _FAMILY_GEOMETRY:
        raise ValueError(f"Unsupported Stage-52 family: {family}")
    profile = dict(_FAMILY_GEOMETRY[family])
    scale = float(_TIMEFRAME_SCALE.get(str(timeframe), 1.0))
    stop_pct = float(profile["stop_pct"]) * scale
    first_target_pct = float(profile["first_target_pct"]) * scale
    stretch_target_pct = float(profile["stretch_target_pct"]) * scale
    entry_price = float(max(entry_reference_price, 1e-6))
    zone_half_width = stop_pct * 0.25
    return {
        "entry_zone": {
            "low": float(round(entry_price * (1.0 - zone_half_width), 8)),
            "high": float(round(entry_price * (1.0 + zone_half_width), 8)),
        },
        "invalidation_event": str(profile["stop_logic"]),
        "stop_distance_pct": float(round(stop_pct, 8)),
        "first_target_pct": float(round(first_target_pct, 8)),
        "stretch_target_pct": float(round(stretch_target_pct, 8)),
        "expected_hold_bars": int(max(2, round(float(profile["hold_bars"]) * scale))),
        "estimated_round_trip_cost_pct": float(round(round_trip_cost_pct, 8)),
        "entry_logic": str(profile["entry_logic"]),
        "stop_logic": str(profile["stop_logic"]),
        "target_logic": str(profile["target_logic"]),
        "hold_logic": str(profile["hold_logic"]),
    }


def _pre_replay_reject_reason(
    *,
    family: str,
    timeframe: str,
    trigger: str,
    confirmation: str,
    invalidation: str,
    rr_model: dict[str, Any],
    cost_edge_proxy: float,
) -> str:
    if family not in set(ALLOWED_SETUP_FAMILIES) or timeframe not in set(ALLOWED_DISCOVERY_TIMEFRAMES):
        return "REJECT::TIMEFRAME_MISMATCH"
    if not str(trigger).strip():
        return "REJECT::WEAK_TRIGGER"
    if not str(confirmation).strip():
        return "REJECT::NO_CONFIRMATION"
    if not str(invalidation).strip():
        return "REJECT::BAD_GEOMETRY"
    if float(rr_model.get("first_target_rr", 0.0)) < 1.5:
        return "REJECT::BAD_GEOMETRY"
    if float(cost_edge_proxy) <= 0.0:
        return "REJECT::COST_MARGIN_TOO_LOW"
    return ""
