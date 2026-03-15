from __future__ import annotations

from buffmini.research.usefulness import (
    build_family_replay_death_map,
    compare_usefulness,
    identify_dead_weight_families,
)


def test_stage95_usefulness_delta_and_dead_weight_detection() -> None:
    before_rows = [
        {
            "candidate_id": "a",
            "family": "structure_pullback_continuation",
            "candidate_hierarchy": "junk",
            "final_class": "rejected",
            "rank_score": 0.20,
            "near_miss_distance": 1.2,
            "first_death_stage": "replay",
        },
        {
            "candidate_id": "b",
            "family": "liquidity_sweep_reversal",
            "candidate_hierarchy": "interesting_but_fragile",
            "final_class": "promising_but_unproven",
            "rank_score": 0.33,
            "near_miss_distance": 0.8,
            "first_death_stage": "walkforward",
        },
    ]
    after_rows = [
        {
            "candidate_id": "a",
            "family": "structure_pullback_continuation",
            "candidate_hierarchy": "interesting_but_fragile",
            "final_class": "promising_but_unproven",
            "rank_score": 0.42,
            "near_miss_distance": 0.7,
            "first_death_stage": "walkforward",
        },
        {
            "candidate_id": "c",
            "family": "failed_breakout_reversal",
            "candidate_hierarchy": "junk",
            "final_class": "rejected",
            "rank_score": 0.10,
            "near_miss_distance": 1.8,
            "first_death_stage": "replay",
        },
    ]
    comparison = compare_usefulness(before_rows=before_rows, after_rows=after_rows)
    assert comparison["delta"]["useful_candidate_delta"] == 0
    assert comparison["delta"]["promising_delta"] == 0
    replay_map = build_family_replay_death_map(after_rows)
    assert any(row["family"] == "failed_breakout_reversal" and row["replay_deaths"] == 1 for row in replay_map)
    dead_weight = identify_dead_weight_families(
        [
            {
                "family": "failed_breakout_reversal",
                "after_useful_candidate_count": 0,
                "after_candidate_count": 6,
                "after_replay_death_fraction": 1.0,
                "after_near_miss_count": 0,
            }
        ]
    )
    assert dead_weight[0]["family"] == "failed_breakout_reversal"
