"""Validation utilities."""

from buffmini.validation.evidence import (
    ALLOWED_METRIC_SOURCE_TYPES,
    ALLOWED_STAGE_ROLES,
    DECISION_BLOCKED_SOURCE_TYPES,
    REAL_DECISION_SOURCE_TYPES,
    build_metric_evidence,
    decision_evidence_guard,
    stage_role_from_source,
    validate_metric_evidence,
    validate_metric_evidence_batch,
)
from buffmini.validation.leakage_harness import run_registered_features_harness


def resolve_validation_candidate(*args, **kwargs):
    from buffmini.validation.candidate_runtime import resolve_validation_candidate as _impl

    return _impl(*args, **kwargs)


def hydrate_candidate_record(*args, **kwargs):
    from buffmini.validation.candidate_runtime import hydrate_candidate_record as _impl

    return _impl(*args, **kwargs)


def load_candidate_market_frame(*args, **kwargs):
    from buffmini.validation.candidate_runtime import load_candidate_market_frame as _impl

    return _impl(*args, **kwargs)


def run_candidate_replay(*args, **kwargs):
    from buffmini.validation.candidate_runtime import run_candidate_replay as _impl

    return _impl(*args, **kwargs)


def evaluate_candidate_walkforward(*args, **kwargs):
    from buffmini.validation.candidate_runtime import evaluate_candidate_walkforward as _impl

    return _impl(*args, **kwargs)


def estimate_trade_monte_carlo(*args, **kwargs):
    from buffmini.validation.candidate_runtime import estimate_trade_monte_carlo as _impl

    return _impl(*args, **kwargs)


def evaluate_cross_perturbation(*args, **kwargs):
    from buffmini.validation.candidate_runtime import evaluate_cross_perturbation as _impl

    return _impl(*args, **kwargs)


def compute_transfer_metrics(*args, **kwargs):
    from buffmini.validation.candidate_runtime import compute_transfer_metrics as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "ALLOWED_METRIC_SOURCE_TYPES",
    "ALLOWED_STAGE_ROLES",
    "DECISION_BLOCKED_SOURCE_TYPES",
    "REAL_DECISION_SOURCE_TYPES",
    "build_metric_evidence",
    "compute_transfer_metrics",
    "decision_evidence_guard",
    "estimate_trade_monte_carlo",
    "evaluate_candidate_walkforward",
    "evaluate_cross_perturbation",
    "hydrate_candidate_record",
    "load_candidate_market_frame",
    "resolve_validation_candidate",
    "run_registered_features_harness",
    "run_candidate_replay",
    "stage_role_from_source",
    "validate_metric_evidence",
    "validate_metric_evidence_batch",
]
