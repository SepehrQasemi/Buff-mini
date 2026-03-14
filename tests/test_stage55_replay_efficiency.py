from __future__ import annotations

import pandas as pd

from buffmini.stage55 import allocate_replay_budget, build_replay_cache_key, validate_phase_timings


def test_stage55_cache_key_is_stable() -> None:
    key1 = build_replay_cache_key(data_hash="a", setup_signature="b", timeframe="1h", cost_model="simple", scope_id="scope")
    key2 = build_replay_cache_key(data_hash="a", setup_signature="b", timeframe="1h", cost_model="simple", scope_id="scope")
    assert key1 == key2


def test_stage55_budget_allocation_accepts_config_keys() -> None:
    candidates = pd.DataFrame([{"candidate_id": f"c{idx}", "replay_priority": 1.0 - idx * 0.1} for idx in range(10)])
    allocation = allocate_replay_budget(
        candidates,
        budget={
            "candidate_limit": 6,
            "stage_a_limit": 4,
            "stage_b_limit": 2,
            "micro_replay_limit": 3,
            "full_replay_limit": 2,
            "walkforward_limit": 1,
            "monte_carlo_limit": 1,
        },
    )
    validate_phase_timings(
        {
            "candidate_generation": 0.1,
            "stage_a_gate": 0.1,
            "stage_b_gate": 0.1,
            "micro_replay": 0.1,
            "full_replay": 0.1,
            "walkforward": 0.1,
            "monte_carlo": 0.1,
        }
    )
    assert allocation["counts"]["precheck"] == 6
    assert allocation["counts"]["full_replay"] == 2
