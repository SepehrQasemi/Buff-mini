"""Stage-52 setup candidate v2 exports."""

from .setup_v2 import (
    ALLOWED_SETUP_FAMILIES,
    build_setup_candidate_v2,
    build_economic_fingerprint,
    deduplicate_setup_candidates_by_economics,
    evaluate_family_coverage,
    summarize_stage52_candidates,
    upgrade_candidates_to_v2,
    validate_setup_candidate_v2,
)

__all__ = [
    "ALLOWED_SETUP_FAMILIES",
    "build_setup_candidate_v2",
    "build_economic_fingerprint",
    "deduplicate_setup_candidates_by_economics",
    "evaluate_family_coverage",
    "summarize_stage52_candidates",
    "upgrade_candidates_to_v2",
    "validate_setup_candidate_v2",
]
