"""Stage-98 mechanism saturation reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from buffmini.constants import PROJECT_ROOT
from buffmini.research.mechanisms import generate_mechanism_source_candidates, mechanism_registry
from buffmini.stage51.scope import resolve_research_scope
from buffmini.stage70.search_expansion import collapse_similarity_candidates, compress_subfamily_hypotheses, deduplicate_economic_candidates
from buffmini.utils.hashing import stable_hash


def evaluate_mechanism_saturation(config: dict[str, Any]) -> dict[str, Any]:
    scope = resolve_research_scope(config)
    families = list(scope.get("available_setup_families") or [])
    timeframes = [value for value in (scope.get("discovery_timeframes") or []) if value in {"15m", "30m", "1h", "4h"}]
    registry = mechanism_registry()
    family_rows = []
    for row in registry:
        family_rows.append(
            {
                "family": str(row.get("family", "")),
                "subfamily_count": int(len(list(row.get("subfamilies", [])))),
                "context_count": int(len(list(row.get("contexts", [])))),
                "trigger_count": int(len(list(row.get("triggers", [])))),
                "confirmation_count": int(len(list(row.get("confirmations", [])))),
                "participation_count": int(len(list(row.get("participation_styles", [])))),
                "invalidation_count": int(len(list(row.get("invalidations", [])))),
                "exit_family_count": int(len(list(row.get("exit_families", [])))),
                "time_stop_count": int(len(list(row.get("time_stops", [])))),
                "regime_count": int(len(list(row.get("expected_regimes", [])))),
            }
        )
    raw = generate_mechanism_source_candidates(
        discovery_timeframes=list(timeframes),
        budget_mode_selected="full_audit",
        active_families=families,
        target_min_candidates=1500,
    )
    compressed = compress_subfamily_hypotheses(raw, max_subfamilies_per_family=4, max_variants_per_subfamily=256)
    collapsed = collapse_similarity_candidates(compressed, max_per_bucket=4)
    deduped = deduplicate_economic_candidates(collapsed)
    prior = _load_prior_stage87_summary()
    raw_count = int(len(raw))
    compressed_count = int(len(compressed))
    deduped_count = int(len(deduped))
    precompression_duplication_ratio = float(round(1.0 - (compressed_count / max(1, raw_count)), 6))
    trivial_duplication_ratio = float(round(1.0 - (deduped_count / max(1, compressed_count)), 6))
    stage98b_required = bool(precompression_duplication_ratio > 0.88)
    return {
        "families": families,
        "timeframes": list(timeframes),
        "family_rows": family_rows,
        "raw_candidate_count": raw_count,
        "post_compression_candidate_count": compressed_count,
        "post_similarity_collapse_count": int(len(collapsed)),
        "post_dedup_candidate_count": deduped_count,
        "precompression_duplication_ratio": precompression_duplication_ratio,
        "trivial_duplication_ratio": trivial_duplication_ratio,
        "stage87_reference": prior,
        "richness_delta": {
            "raw_candidate_delta_vs_stage87": int(raw_count - int(prior.get("raw_candidate_count", 0))),
            "dedup_candidate_delta_vs_stage87": int(deduped_count - int(prior.get("post_similarity_collapse_count", 0))),
        },
        "stage98b_required": stage98b_required,
        "stage98b_applied": True,
        "stage98b_reason": "duplication_ratio_too_high" if stage98b_required else "",
        "summary_hash": stable_hash(
            {
                "raw_candidate_count": raw_count,
                "post_compression_candidate_count": compressed_count,
                "post_dedup_candidate_count": deduped_count,
                "trivial_duplication_ratio": trivial_duplication_ratio,
            },
            length=16,
        ),
    }


def _load_prior_stage87_summary() -> dict[str, Any]:
    path = PROJECT_ROOT / "docs" / "stage87_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
