"""Research execution helpers for controlled evaluation and campaign work."""

from .behavior import build_behavioral_fingerprints
from .diagnostics import classify_candidate_tier, compute_candidate_risk_card
from .modes import STANDARD_RUN_TYPES, build_mode_context, resolve_run_mode


def evaluate_detectability_suite(*args, **kwargs):
    from .synthetic_lab import evaluate_detectability_suite as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "STANDARD_RUN_TYPES",
    "build_behavioral_fingerprints",
    "build_mode_context",
    "classify_candidate_tier",
    "compute_candidate_risk_card",
    "evaluate_detectability_suite",
    "resolve_run_mode",
]
