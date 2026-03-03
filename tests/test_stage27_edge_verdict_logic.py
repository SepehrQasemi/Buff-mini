from __future__ import annotations

import pandas as pd

from buffmini.stage27.edge_verdict import contextual_edge_verdict


def test_contextual_edge_verdict_classifies_robust_weak_noise() -> None:
    rows = []
    # Robust group: 5 windows, 4 positive.
    for i, val in enumerate([0.2, 0.1, 0.3, 0.15, -0.05]):
        rows.append(
            {
                "best_context": "TREND",
                "best_rulelet": "TrendPullback",
                "timeframe": "1h",
                "window_index": i,
                "best_exp_lcb": val,
                "best_trades_in_context": 40,
            }
        )
    # Weak group: sparse positives.
    for i, val in enumerate([0.05, 0.04, -0.01, -0.02]):
        rows.append(
            {
                "best_context": "RANGE",
                "best_rulelet": "RangeFade",
                "timeframe": "1h",
                "window_index": i,
                "best_exp_lcb": val,
                "best_trades_in_context": 20,
            }
        )
    # Noise group.
    for i, val in enumerate([-0.2, -0.1, -0.05, -0.03]):
        rows.append(
            {
                "best_context": "CHOP",
                "best_rulelet": "ChopFilterGate",
                "timeframe": "1h",
                "window_index": i,
                "best_exp_lcb": val,
                "best_trades_in_context": 10,
            }
        )
    out = contextual_edge_verdict(pd.DataFrame(rows), min_windows=3, robust_repeat_rate=0.6)
    mapping = {(row["context"], row["rulelet"]): row["verdict"] for row in out["rows"]}
    assert mapping[("TREND", "TrendPullback")] == "ROBUST_IN_CONTEXT"
    assert mapping[("RANGE", "RangeFade")] == "WEAK"
    assert mapping[("CHOP", "ChopFilterGate")] == "NOISE"
    assert out["has_contextual_edge"] is True


def test_contextual_edge_verdict_no_edge_case() -> None:
    df = pd.DataFrame(
        {
            "best_context": ["RANGE", "RANGE", "TREND"],
            "best_rulelet": ["RangeFade", "RangeFade", "TrendPullback"],
            "timeframe": ["1h", "1h", "1h"],
            "best_exp_lcb": [-0.1, -0.2, -0.05],
            "best_trades_in_context": [20, 20, 15],
        }
    )
    out = contextual_edge_verdict(df)
    assert out["has_contextual_edge"] is False
    assert out["policy_verdict"] == "NO_CONTEXTUAL_EDGE"
