"""Stage-20 robust objective with constraints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveConstraints:
    min_tpm: float = 5.0
    max_tpm: float = 80.0
    exposure_min: float = 0.01
    max_dd_p95: float = 0.30
    max_drag_penalty: float = 5.0
    min_horizon_consistency: float = 0.20


def robust_objective(
    *,
    exp_lcb: float,
    tpm: float,
    exposure_ratio: float,
    max_dd_p95: float,
    drag_penalty: float,
    horizon_consistency: float,
    constraints: ObjectiveConstraints,
) -> dict[str, float | bool | str]:
    """Score candidate under hard constraints."""

    reasons: list[str] = []
    valid = True
    if tpm < constraints.min_tpm or tpm > constraints.max_tpm:
        valid = False
        reasons.append("TPM_OUT_OF_BOUNDS")
    if exposure_ratio < constraints.exposure_min:
        valid = False
        reasons.append("LOW_EXPOSURE")
    if max_dd_p95 > constraints.max_dd_p95:
        valid = False
        reasons.append("MAX_DD_BREACH")
    if drag_penalty > constraints.max_drag_penalty:
        valid = False
        reasons.append("DRAG_TOO_HIGH")
    if horizon_consistency < constraints.min_horizon_consistency:
        valid = False
        reasons.append("LOW_HORIZON_CONSISTENCY")

    score = float(exp_lcb - drag_penalty - max(0.0, max_dd_p95 - constraints.max_dd_p95))
    if not valid:
        score -= 1_000_000.0
    return {
        "score": score,
        "valid": valid,
        "reason": "|".join(reasons) if reasons else "VALID",
    }

