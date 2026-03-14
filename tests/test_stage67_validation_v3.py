from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage67 import build_anchored_splits, evaluate_validation_protocol_v3


def test_stage67_builds_splits_and_evaluates() -> None:
    splits = build_anchored_splits(n_rows=500, n_splits=4, min_train=128, test_size=64, purge_gap=8)
    assert len(splits) >= 1
    frame = pd.DataFrame(
        {
            "expected_net_after_cost_label": np.linspace(-0.001, 0.003, 500),
            "tp_before_sl_label": np.where(np.arange(500) % 2 == 0, 1.0, 0.0),
        }
    )
    out = evaluate_validation_protocol_v3(
        dataset=frame,
        score_column="expected_net_after_cost_label",
        label_column="tp_before_sl_label",
        stage_a_survivors=5,
        stage_b_survivors=3,
    )
    assert out["split_count"] >= 1
    assert out["status"] in {"SUCCESS", "PARTIAL"}

