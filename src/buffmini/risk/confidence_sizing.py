"""Stage-6 confidence-weighted position sizing helpers."""

from __future__ import annotations

import math
from typing import Any


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(float(low), min(float(high), float(value))))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-float(value))
        return float(1.0 / (1.0 + z))
    z = math.exp(float(value))
    return float(z / (1.0 + z))


def candidate_confidence(
    exp_lcb_holdout: float,
    pf_adj_holdout: float,
    scale: float,
) -> float:
    """Map candidate quality metrics into confidence in [0, 1]."""

    scale_value = float(scale)
    if scale_value <= 0:
        raise ValueError("scale must be > 0")

    edge_term = _sigmoid(float(exp_lcb_holdout) / scale_value)
    pf_term = _clamp(float(pf_adj_holdout) / 2.0, 0.0, 1.0)
    return _clamp(edge_term * pf_term, 0.0, 1.0)


def confidence_multiplier(
    confidence: float,
    lower: float = 0.5,
    upper: float = 1.5,
) -> float:
    """Convert confidence to a deterministic sizing multiplier."""

    return _clamp(0.5 + float(confidence), float(lower), float(upper))


def renormalize_signed_weights(
    signed_weights: dict[str, float],
    max_abs_sum: float,
) -> tuple[dict[str, float], float]:
    """Scale signed component weights so sum(abs(weight)) <= max_abs_sum."""

    limit = float(max_abs_sum)
    if limit <= 0:
        return {key: 0.0 for key in signed_weights}, 0.0
    abs_sum = float(sum(abs(float(value)) for value in signed_weights.values()))
    if abs_sum <= limit or abs_sum <= 0:
        return {key: float(value) for key, value in signed_weights.items()}, 1.0
    scale = float(limit / abs_sum)
    return {key: float(value) * scale for key, value in signed_weights.items()}, scale


def extract_candidate_metric(
    candidate_meta: dict[str, Any] | None,
    key: str,
    fallback: float = 0.0,
) -> float:
    """Read candidate metric safely from metadata payload."""

    if not isinstance(candidate_meta, dict):
        return float(fallback)
    try:
        value = candidate_meta.get(key, fallback)
        if value is None:
            return float(fallback)
        return float(value)
    except Exception:
        return float(fallback)

