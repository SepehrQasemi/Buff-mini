from __future__ import annotations

from buffmini.alpha_v2.objective import ObjectiveConstraints, robust_objective


def test_stage20_objective_enforces_constraints() -> None:
    constraints = ObjectiveConstraints(min_tpm=5.0, max_tpm=80.0, exposure_min=0.02, max_dd_p95=0.3, max_drag_penalty=2.0)
    bad = robust_objective(
        exp_lcb=1.0,
        tpm=2.0,
        exposure_ratio=0.01,
        max_dd_p95=0.4,
        drag_penalty=3.0,
        horizon_consistency=0.1,
        constraints=constraints,
    )
    assert not bool(bad["valid"])
    assert float(bad["score"]) < -1_000.0


def test_stage20_objective_accepts_valid_candidate() -> None:
    constraints = ObjectiveConstraints(min_tpm=1.0, max_tpm=80.0, exposure_min=0.01, max_dd_p95=0.4, max_drag_penalty=5.0)
    good = robust_objective(
        exp_lcb=2.5,
        tpm=10.0,
        exposure_ratio=0.20,
        max_dd_p95=0.15,
        drag_penalty=0.5,
        horizon_consistency=0.6,
        constraints=constraints,
    )
    assert bool(good["valid"])
    assert str(good["reason"]) == "VALID"
    assert float(good["score"]) > 0.0


def test_stage20_objective_is_deterministic() -> None:
    constraints = ObjectiveConstraints()
    a = robust_objective(
        exp_lcb=0.8,
        tpm=12.0,
        exposure_ratio=0.10,
        max_dd_p95=0.2,
        drag_penalty=0.4,
        horizon_consistency=0.5,
        constraints=constraints,
    )
    b = robust_objective(
        exp_lcb=0.8,
        tpm=12.0,
        exposure_ratio=0.10,
        max_dd_p95=0.2,
        drag_penalty=0.4,
        horizon_consistency=0.5,
        constraints=constraints,
    )
    assert a == b

