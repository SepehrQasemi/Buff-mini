from __future__ import annotations

import pandas as pd

from buffmini.stage48.tradability_learning import score_candidates_with_ranker


def test_stage48_ranker_outputs_expected_columns_and_determinism() -> None:
    candidates = pd.DataFrame(
        [
            {"candidate_id": "a", "beam_score": 0.4, "exp_lcb_proxy": 0.01},
            {"candidate_id": "b", "beam_score": 0.6, "exp_lcb_proxy": 0.00},
        ]
    )
    labels = pd.DataFrame(
        {
            "tradable": [1, 0, 1, 1],
            "net_return_after_cost": [0.01, -0.002, 0.005, 0.003],
            "rr_adequacy": [1, 1, 1, 0],
        }
    )
    one = score_candidates_with_ranker(candidates, labels)
    two = score_candidates_with_ranker(candidates, labels)
    for key in ("candidate_id", "rank_score", "predicted_tradability", "replay_worthiness"):
        assert key in one.columns
    assert one["candidate_id"].tolist() == two["candidate_id"].tolist()

