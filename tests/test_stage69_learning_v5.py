from __future__ import annotations

import pandas as pd

from buffmini.stage69 import build_campaign_memory_rows_v5, derive_campaign_priors_v5


def test_stage69_builds_memory_and_priors() -> None:
    outcomes = pd.DataFrame(
        [
            {"candidate_id": "c1", "family": "f1", "timeframe": "1h", "cost_edge_proxy": 0.001, "tp_before_sl_label": 1.0, "expected_net_after_cost_label": 0.001},
            {"candidate_id": "c2", "family": "f2", "timeframe": "2h", "cost_edge_proxy": -0.001, "tp_before_sl_label": 0.0, "expected_net_after_cost_label": -0.001},
        ]
    )
    gated = pd.DataFrame([{"candidate_id": "c1"}])
    rows = build_campaign_memory_rows_v5(outcomes=outcomes, gated_candidates=gated, stage28_run_id="r1")
    priors = derive_campaign_priors_v5(rows)
    assert len(rows) == 2
    assert "family_allocation" in priors
    assert "threshold_prior" in priors

