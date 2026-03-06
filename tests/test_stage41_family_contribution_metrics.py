from __future__ import annotations

import pandas as pd

from buffmini.stage41.contribution import compute_family_contribution_metrics


def test_stage41_family_contribution_metrics_schema_and_values() -> None:
    layer_a = pd.DataFrame(
        [
            {"family": "funding"},
            {"family": "funding"},
            {"family": "flow"},
            {"family": "taker_buy_sell"},
        ]
    )
    layer_c = pd.DataFrame(
        [
            {"family": "funding"},
            {"family": "flow"},
            {"family": "taker_buy_sell"},
        ]
    )
    stage_a = pd.DataFrame([{"family": "funding"}, {"family": "flow"}])
    stage_b = pd.DataFrame([{"family": "funding"}])
    rows = compute_family_contribution_metrics(
        layer_a=layer_a,
        layer_c=layer_c,
        stage_a_survivors=stage_a,
        stage_b_survivors=stage_b,
        families=["funding", "flow", "taker_buy_sell"],
    )
    assert len(rows) == 3
    funding = next(row for row in rows if row["family"] == "funding")
    assert funding["layer_a_count"] == 2
    assert funding["stage_b_count"] == 1
    for row in rows:
        for key in (
            "candidate_lift",
            "activation_lift",
            "tradability_lift",
            "shortlist_share",
            "final_policy_share",
        ):
            assert key in row

