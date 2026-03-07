from __future__ import annotations

import pandas as pd

from buffmini.stage48.tradability_learning import Stage48Config, route_stage_a_stage_b


def test_stage48_stage_a_stage_b_accounting_consistent() -> None:
    candidates = pd.DataFrame(
        [
            {"candidate_id": "a", "beam_score": 0.8, "exp_lcb_proxy": 0.02},
            {"candidate_id": "b", "beam_score": 0.4, "exp_lcb_proxy": 0.01},
            {"candidate_id": "c", "beam_score": 0.2, "exp_lcb_proxy": -0.01},
        ]
    )
    labels = pd.DataFrame(
        {
            "tradable": [1, 1, 0, 1],
            "net_return_after_cost": [0.01, 0.004, -0.002, 0.002],
            "rr_adequacy": [1, 1, 1, 0],
            "expected_hold_validity": [1, 1, 1, 1],
        }
    )
    out = route_stage_a_stage_b(candidates, labels=labels, cfg=Stage48Config(stage_a_threshold=0.2, stage_b_threshold=0.0))
    counts = dict(out["counts"])
    assert int(counts["stage_b"]) <= int(counts["stage_a"]) <= int(counts["input"])
    assert int(out["strict_direct_survivors_before"]) <= int(counts["input"])

