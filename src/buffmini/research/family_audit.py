"""Family coverage audit and blind-spot analysis for mechanism search."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import pandas as pd

from buffmini.research.mechanisms import mechanism_registry
from buffmini.stage51.scope import resolve_research_scope
from buffmini.stage70.search_expansion import generate_expanded_candidates, similarity_bucket
from buffmini.utils.hashing import stable_hash


FAMILY_GROUPS: dict[str, tuple[str, ...]] = {
    "continuation": ("structure_pullback_continuation", "volatility_regime_transition"),
    "reversal": (
        "liquidity_sweep_reversal",
        "failed_breakout_reversal",
        "exhaustion_mean_reversion",
        "funding_oi_imbalance_reversion",
    ),
    "breakout_compression": ("squeeze_flow_breakout", "volatility_regime_transition"),
    "imbalance_crowding": ("funding_oi_imbalance_reversion", "liquidity_sweep_reversal"),
    "mtf_interaction": ("multi_tf_disagreement_repair", "structure_pullback_continuation"),
    "regime_transition": ("volatility_regime_transition", "multi_tf_disagreement_repair"),
}


def evaluate_family_audit(config: dict[str, Any]) -> dict[str, Any]:
    scope = resolve_research_scope(config)
    families = list(scope.get("available_setup_families", []))
    discovery_timeframes = list(scope.get("discovery_timeframes", []))
    registry_rows = mechanism_registry()
    registry_by_family = {str(row["family"]): row for row in registry_rows}
    inventory = [_family_inventory_row(registry_by_family[str(family)]) for family in families if str(family) in registry_by_family]

    expanded = generate_expanded_candidates(
        discovery_timeframes=discovery_timeframes,
        budget_mode_selected=str((config.get("budget_mode", {}) or {}).get("selected", "search")),
        active_families=families,
        min_search_candidates=800,
    )
    overlap = _build_overlap_analysis(expanded)
    blind_spots = _build_blind_spot_list(inventory)
    return {
        "family_inventory": inventory,
        "overlap_analysis": overlap,
        "blind_spots": blind_spots,
        "family_count": int(len(inventory)),
        "subfamily_total": int(sum(int(row.get("subfamily_count", 0)) for row in inventory)),
        "summary_hash": stable_hash(
            {
                "families": [row.get("family", "") for row in inventory],
                "blind_spots": blind_spots,
                "semantic_overlap_pairs": overlap.get("semantic_overlap_pairs", [])[:10],
            },
            length=16,
        ),
    }


def _family_inventory_row(row: dict[str, Any]) -> dict[str, Any]:
    subfamilies = list(row.get("subfamilies", []))
    contexts = list(row.get("contexts", []))
    triggers = list(row.get("triggers", []))
    confirmations = list(row.get("confirmations", []))
    invalidations = list(row.get("invalidations", []))
    exit_families = list(row.get("exit_families", []))
    time_stops = list(row.get("time_stops", []))
    expected_regimes = list(row.get("expected_regimes", []))
    modules = list(row.get("modules", []))
    return {
        "family": str(row.get("family", "")),
        "subfamily_count": int(row.get("subfamily_count", len(subfamilies))),
        "submechanisms_covered": subfamilies,
        "submechanisms_missing": _infer_missing_submechanisms(row),
        "context_richness": int(len(contexts)),
        "trigger_richness": int(len(triggers)),
        "confirmation_richness": int(len(confirmations)),
        "invalidation_richness": int(len(invalidations)),
        "exit_richness": int(len(exit_families)),
        "hold_horizon_richness": int(len(time_stops)),
        "mtf_richness": int(sum(1 for mod in modules if "mtf" in str(mod).lower())),
        "regime_conditioning_richness": int(len(expected_regimes)),
        "transfer_expectation": str(row.get("transfer_expectation", "")),
        "trade_density_expectation": str(row.get("trade_density_expectation", "")),
        "expected_failure_modes": list(row.get("expected_failure_modes", [])),
        "registry_rationale": _registry_rationale(row),
    }


def _infer_missing_submechanisms(row: dict[str, Any]) -> list[str]:
    family = str(row.get("family", ""))
    covered = set(str(item) for item in row.get("subfamilies", []))
    expected: dict[str, tuple[str, ...]] = {
        "structure_pullback_continuation": ("trend_reset", "breakout_retest"),
        "liquidity_sweep_reversal": ("equal_high_sweep", "equal_low_sweep"),
        "squeeze_flow_breakout": ("nested_compression", "failed_squeeze_rebreak"),
        "failed_breakout_reversal": ("range_break_fail", "trend_break_fail"),
        "exhaustion_mean_reversion": ("multi_bar_climax", "funding_stretch"),
        "funding_oi_imbalance_reversion": ("oi_flush", "funding_spike"),
        "volatility_regime_transition": ("vol_crush_reexpand", "atr_reacceleration"),
        "multi_tf_disagreement_repair": ("htf_flip_realign", "ltf_whipsaw_repair"),
    }
    return [item for item in expected.get(family, ()) if item not in covered]


def _registry_rationale(row: dict[str, Any]) -> str:
    family = str(row.get("family", ""))
    regimes = "/".join(str(item) for item in row.get("expected_regimes", []))
    failures = ",".join(str(item) for item in row.get("expected_failure_modes", [])[:2])
    return f"{family} is designed for {regimes} with expected weak points in {failures}" if regimes else family


def _build_overlap_analysis(expanded: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(expanded, pd.DataFrame) or expanded.empty:
        return {
            "semantic_overlap_pairs": [],
            "behavioral_overlap_pairs": [],
            "fingerprint_overlap_pairs": [],
        }

    work = expanded.copy()
    work["similarity_bucket"] = [similarity_bucket(dict(row)) for row in work.to_dict(orient="records")]
    family_groups = {str(family): frame.reset_index(drop=True) for family, frame in work.groupby("family", dropna=False)}

    semantic_rows: list[dict[str, Any]] = []
    behavioral_rows: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []
    for left_family, right_family in combinations(sorted(family_groups), 2):
        left = family_groups[left_family]
        right = family_groups[right_family]
        semantic_rows.append(
            {
                "family_left": left_family,
                "family_right": right_family,
                "semantic_overlap": float(round(_semantic_overlap(left, right), 6)),
            }
        )
        behavioral_rows.append(
            {
                "family_left": left_family,
                "family_right": right_family,
                "behavioral_overlap": float(round(_behavioral_overlap(left, right), 6)),
            }
        )
        fingerprint_rows.append(
            {
                "family_left": left_family,
                "family_right": right_family,
                "fingerprint_overlap": float(round(_fingerprint_overlap(left, right), 6)),
            }
        )
    return {
        "semantic_overlap_pairs": sorted(semantic_rows, key=lambda row: (-float(row["semantic_overlap"]), row["family_left"], row["family_right"])),
        "behavioral_overlap_pairs": sorted(behavioral_rows, key=lambda row: (-float(row["behavioral_overlap"]), row["family_left"], row["family_right"])),
        "fingerprint_overlap_pairs": sorted(fingerprint_rows, key=lambda row: (-float(row["fingerprint_overlap"]), row["family_left"], row["family_right"])),
    }


def _semantic_overlap(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left_signatures = {
        (
            str(row.get("context", "")),
            str(row.get("trigger", "")),
            str(row.get("confirmation", "")),
            str(row.get("expected_regime", "")),
        )
        for row in left.to_dict(orient="records")
    }
    right_signatures = {
        (
            str(row.get("context", "")),
            str(row.get("trigger", "")),
            str(row.get("confirmation", "")),
            str(row.get("expected_regime", "")),
        )
        for row in right.to_dict(orient="records")
    }
    union = left_signatures | right_signatures
    if not union:
        return 0.0
    return float(len(left_signatures & right_signatures) / len(union))


def _behavioral_overlap(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left_modes = defaultdict(int)
    right_modes = defaultdict(int)
    for row in left.to_dict(orient="records"):
        key = (
            str(row.get("participation_style", row.get("participation", ""))),
            str(row.get("exit_family", "")),
            str(row.get("trade_density_expectation", "")),
        )
        left_modes[key] += 1
    for row in right.to_dict(orient="records"):
        key = (
            str(row.get("participation_style", row.get("participation", ""))),
            str(row.get("exit_family", "")),
            str(row.get("trade_density_expectation", "")),
        )
        right_modes[key] += 1
    keys = set(left_modes) | set(right_modes)
    if not keys:
        return 0.0
    shared = 0.0
    total = 0.0
    for key in keys:
        shared += min(float(left_modes.get(key, 0)), float(right_modes.get(key, 0)))
        total += max(float(left_modes.get(key, 0)), float(right_modes.get(key, 0)))
    return float(shared / max(total, 1.0))


def _fingerprint_overlap(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left_buckets = set(left.get("similarity_bucket", pd.Series(dtype=str)).astype(str))
    right_buckets = set(right.get("similarity_bucket", pd.Series(dtype=str)).astype(str))
    union = left_buckets | right_buckets
    if not union:
        return 0.0
    return float(len(left_buckets & right_buckets) / len(union))


def _build_blind_spot_list(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    present = {str(row.get("family", "")) for row in inventory}
    richness_rows: list[dict[str, Any]] = []
    for theme, expected_families in FAMILY_GROUPS.items():
        available = [family for family in expected_families if family in present]
        richness_rows.append(
            {
                "theme": theme,
                "available_families": available,
                "coverage_ratio": float(round(len(available) / max(1, len(expected_families)), 6)),
                "blind_spot": len(available) < len(expected_families),
            }
        )
    low_richness = [
        {
            "family": str(row.get("family", "")),
            "issue": "mechanism_depth_low",
            "details": {
                "context_richness": int(row.get("context_richness", 0)),
                "trigger_richness": int(row.get("trigger_richness", 0)),
                "confirmation_richness": int(row.get("confirmation_richness", 0)),
                "submechanisms_missing": list(row.get("submechanisms_missing", [])),
            },
        }
        for row in inventory
        if int(row.get("context_richness", 0)) < 2
        or int(row.get("trigger_richness", 0)) < 2
        or int(row.get("confirmation_richness", 0)) < 2
        or len(list(row.get("submechanisms_missing", []))) > 0
    ]
    theme_blind_spots = [row for row in richness_rows if bool(row.get("blind_spot", False))]
    return theme_blind_spots + low_richness
