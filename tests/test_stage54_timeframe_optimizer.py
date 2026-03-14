from __future__ import annotations

import pandas as pd

from buffmini.stage54 import build_timeframe_metrics, select_timeframe_promotions, tpe_suggest


def test_stage54_selects_top_timeframes() -> None:
    frame = pd.DataFrame(
        [
            {"candidate_id": "a", "timeframe": "15m", "tp_before_sl_prob": 0.58, "expected_net_after_cost": 0.001, "rr_model": {"first_target_rr": 1.6}, "cost_edge_proxy": 0.0004, "pre_replay_reject_reason": ""},
            {"candidate_id": "b", "timeframe": "1h", "tp_before_sl_prob": 0.72, "expected_net_after_cost": 0.003, "rr_model": {"first_target_rr": 1.9}, "cost_edge_proxy": 0.0010, "pre_replay_reject_reason": ""},
            {"candidate_id": "c", "timeframe": "4h", "tp_before_sl_prob": 0.54, "expected_net_after_cost": -0.001, "rr_model": {"first_target_rr": 1.4}, "cost_edge_proxy": -0.0002, "pre_replay_reject_reason": "REJECT::COST_MARGIN_TOO_LOW"},
        ]
    )
    metrics = build_timeframe_metrics(frame, runtime_by_timeframe={"15m": 12.0, "1h": 6.0, "4h": 4.0})
    promotion = select_timeframe_promotions(metrics, promotion_timeframes=2, final_validation_timeframes=1)
    assert promotion["promotion_timeframes"][0] == "1h"
    assert promotion["final_validation_timeframes"] == ["1h"]


def test_stage54_tpe_suggestion_is_deterministic() -> None:
    history = pd.DataFrame(
        [
            {"threshold": 0.50, "weight": 0.10, "objective": 0.01},
            {"threshold": 0.55, "weight": 0.15, "objective": 0.03},
            {"threshold": 0.60, "weight": 0.20, "objective": 0.02},
        ]
    )
    suggestion = tpe_suggest(history, search_space={"threshold": [0.50, 0.55, 0.60], "weight": [0.10, 0.15, 0.20]})
    assert suggestion == {"threshold": 0.55, "weight": 0.15}
