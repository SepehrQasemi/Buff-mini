from __future__ import annotations

from buffmini.stage49.self_learning3 import family_module_downweighting


def test_stage49_family_downweighting_penalizes_weak_families() -> None:
    rows = [
        {"family_name": "flow", "exp_lcb": 0.01, "activation_rate": 0.4},
        {"family_name": "flow", "exp_lcb": 0.02, "activation_rate": 0.5},
        {"family_name": "weak", "exp_lcb": -0.03, "activation_rate": 0.1},
        {"family_name": "weak", "exp_lcb": -0.01, "activation_rate": 0.1},
    ]
    weights = family_module_downweighting(rows)
    assert float(weights["flow"]) > float(weights["weak"])

