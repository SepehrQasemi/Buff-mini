from __future__ import annotations

from buffmini.stage11.evaluate import _trade_count_guard


def _summary(trade_count: float, pf: float, exp_lcb: float) -> dict:
    return {
        "baseline_vs_stage10": {
            "stage10": {
                "trade_count": trade_count,
                "profit_factor": pf,
                "exp_lcb": exp_lcb,
            }
        }
    }


def test_stage11_trade_count_guard_fails_without_material_improvement() -> None:
    baseline = _summary(trade_count=1000, pf=1.2, exp_lcb=0.1)
    candidate = _summary(trade_count=700, pf=1.21, exp_lcb=0.2)
    guard = _trade_count_guard(
        baseline=baseline,
        candidate=candidate,
        cfg={"confirm_max_drop_pct": 25.0, "material_pf_improvement": 0.2, "material_exp_lcb_improvement": 0.5},
        bias_enabled=False,
        confirm_enabled=True,
    )
    assert guard["pass"] is False
    assert guard["observed_drop_pct"] > 25.0


def test_stage11_trade_count_guard_passes_with_material_improvement() -> None:
    baseline = _summary(trade_count=1000, pf=1.0, exp_lcb=0.0)
    candidate = _summary(trade_count=700, pf=1.3, exp_lcb=0.8)
    guard = _trade_count_guard(
        baseline=baseline,
        candidate=candidate,
        cfg={"confirm_max_drop_pct": 25.0, "material_pf_improvement": 0.2, "material_exp_lcb_improvement": 0.5},
        bias_enabled=False,
        confirm_enabled=True,
    )
    assert guard["material_improvement"] is True
    assert guard["pass"] is True
