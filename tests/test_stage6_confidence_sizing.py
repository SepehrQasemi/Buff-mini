"""Stage-6 confidence sizing tests."""

from __future__ import annotations

from buffmini.risk.confidence_sizing import (
    candidate_confidence,
    confidence_multiplier,
    renormalize_signed_weights,
)


def test_multiplier_monotonicity() -> None:
    low_conf = candidate_confidence(exp_lcb_holdout=-2.0, pf_adj_holdout=0.8, scale=2.0)
    high_conf = candidate_confidence(exp_lcb_holdout=2.5, pf_adj_holdout=1.6, scale=2.0)
    assert low_conf < high_conf
    assert confidence_multiplier(low_conf) < confidence_multiplier(high_conf)


def test_renormalization_respects_cap() -> None:
    weights = {"a": 0.8, "b": -0.7, "c": 0.5}
    scaled, factor = renormalize_signed_weights(weights, max_abs_sum=1.0)
    assert 0.0 < factor < 1.0
    assert sum(abs(value) for value in scaled.values()) <= 1.0 + 1e-12


def test_confidence_sizing_deterministic() -> None:
    first = candidate_confidence(exp_lcb_holdout=1.25, pf_adj_holdout=1.4, scale=2.0)
    second = candidate_confidence(exp_lcb_holdout=1.25, pf_adj_holdout=1.4, scale=2.0)
    assert first == second

