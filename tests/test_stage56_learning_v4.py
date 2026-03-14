from __future__ import annotations

import pandas as pd

from buffmini.stage56 import expand_registry_rows_v4, mutation_guidance_v4


def test_stage56_registry_v4_rows_have_required_fields() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "family": "structure_pullback_continuation",
                "timeframe": "1h",
                "context": {"primary": "trend"},
                "trigger": {"type": "pullback"},
                "entry_logic": "enter",
                "stop_logic": "stop",
                "target_logic": "target",
                "rr_model": {"first_target_rr": 1.7},
                "pre_replay_reject_reason": "",
                "exp_lcb_proxy": 0.001,
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "expected_net_after_cost": 0.002,
                "mae_pct": -0.003,
                "mfe_pct": 0.008,
                "replay_priority": 0.7,
            }
        ]
    )
    rows = expand_registry_rows_v4(candidates=candidates, predictions=predictions, seed=42, run_id="r1")
    assert rows
    assert rows[0]["mutation_guidance"]


def test_stage56_mutation_order_prefers_geometry_over_cost_when_present() -> None:
    guidance = mutation_guidance_v4(
        {
            "stage_a_failures": ["REJECT::BAD_GEOMETRY", "REJECT::COST_MARGIN_TOO_LOW"],
            "stage_b_failures": [],
        }
    )
    assert guidance == "alter_geometry_and_invalidation"


def test_stage56_registry_drops_nan_pre_replay_reason() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "family": "structure_pullback_continuation",
                "timeframe": "1h",
                "context": {"primary": "trend"},
                "trigger": {"type": "pullback"},
                "entry_logic": "enter",
                "stop_logic": "stop",
                "target_logic": "target",
                "rr_model": {"first_target_rr": 1.7},
                "pre_replay_reject_reason": float("nan"),
                "exp_lcb_proxy": 0.001,
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "expected_net_after_cost": 0.002,
                "mae_pct": -0.003,
                "mfe_pct": 0.008,
                "replay_priority": 0.7,
            }
        ]
    )
    rows = expand_registry_rows_v4(candidates=candidates, predictions=predictions, seed=42, run_id="r1")
    assert rows
    assert "nan" not in [str(v).lower() for v in rows[0]["stage_a_failures"]]
