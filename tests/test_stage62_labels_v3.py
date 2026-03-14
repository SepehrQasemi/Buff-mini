from __future__ import annotations

import pandas as pd

from buffmini.stage62 import build_candidate_outcomes_v3, build_training_dataset_v3, evaluate_quality_gate_v3


def test_stage62_builds_candidate_outcomes_and_quality_gate() -> None:
    stage52 = pd.DataFrame(
        [
            {"candidate_id": "s52_1", "source_candidate_id": "s47_1", "cost_edge_proxy": 0.001, "family": "f1", "timeframe": "1h"},
            {"candidate_id": "s52_2", "source_candidate_id": "s47_2", "cost_edge_proxy": -0.001, "family": "f2", "timeframe": "1h"},
        ]
    )
    stage48_a = pd.DataFrame([{"candidate_id": "s47_1"}])
    stage48_b = pd.DataFrame([{"candidate_id": "s47_1"}])
    pred = pd.DataFrame([{"candidate_id": "s52_1", "tp_before_sl_prob": 0.7, "expected_net_after_cost": 0.002}])
    outcomes = build_candidate_outcomes_v3(
        stage52_candidates=stage52,
        stage48_stage_a=stage48_a,
        stage48_stage_b=stage48_b,
        stage53_predictions=pred,
    )
    features = ["cost_edge_proxy", "tp_before_sl_prob", "expected_net_after_cost"]
    ds = build_training_dataset_v3(outcomes, feature_columns=features)
    quality = evaluate_quality_gate_v3(ds, feature_columns=features)
    assert len(outcomes) == 2
    assert "tp_before_sl_label" in ds.columns
    assert quality["passed"] is False
    assert quality["label_coverage"] >= 0.0

