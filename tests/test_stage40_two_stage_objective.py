from __future__ import annotations

import pandas as pd

from buffmini.stage40.objective import TradabilityConfig, route_two_stage_objective


def test_stage40_two_stage_objective_routes_candidates_and_counts() -> None:
    candidates = pd.DataFrame(
        [
            {"candidate_id": "a", "layer_score": 0.9, "exp_lcb_proxy": 0.02, "broad_context": "trend"},
            {"candidate_id": "b", "layer_score": 0.5, "exp_lcb_proxy": 0.01, "broad_context": "squeeze"},
            {"candidate_id": "c", "layer_score": 0.1, "exp_lcb_proxy": -0.02, "broad_context": "range"},
        ]
    )
    labels = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=10, freq="1h", tz="UTC"),
            "tp_before_sl": [1.0] * 10,
            "net_return_after_cost": [0.002] * 10,
            "mae_pct": [-0.001] * 10,
            "adverse_excursion_ok": [True] * 10,
            "tradable": [1] * 10,
        }
    )
    cfg = TradabilityConfig(stage_a_threshold=0.35, stage_b_threshold=0.0)
    out = route_two_stage_objective(candidates, labels=labels, cfg=cfg)
    counts = dict(out["counts"])
    assert counts["input"] == 3
    assert counts["stage_a"] >= counts["stage_b"] >= 0
    assert counts["before_strict_direct"] >= counts["stage_b"]
    assert out["bottleneck_step"] in {"stage_a_activation", "stage_b_robustness"}

