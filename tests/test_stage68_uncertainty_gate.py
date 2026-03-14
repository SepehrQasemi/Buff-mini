from __future__ import annotations

import pandas as pd

from buffmini.stage68 import apply_uncertainty_gate_v3


def test_stage68_filters_by_confidence_and_net() -> None:
    frame = pd.DataFrame(
        [
            {"candidate_id": "a", "tp_before_sl_prob": 0.70, "expected_net_after_cost": 0.002, "uncertainty_score": 0.10},
            {"candidate_id": "b", "tp_before_sl_prob": 0.60, "expected_net_after_cost": -0.001, "uncertainty_score": 0.05},
            {"candidate_id": "c", "tp_before_sl_prob": 0.58, "expected_net_after_cost": 0.001, "uncertainty_score": 0.40},
        ]
    )
    out = apply_uncertainty_gate_v3(frame, max_uncertainty=0.25)
    assert out["counts"]["input"] == 3
    assert out["counts"]["accepted"] == 1
    assert out["gated"]["candidate_id"].tolist() == ["a"]

