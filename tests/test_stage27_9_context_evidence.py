from __future__ import annotations

import pandas as pd

from buffmini.stage27.context_evidence import classify_context_evidence, evaluate_context_evidence


def test_stage27_9_context_evidence_thresholds() -> None:
    assert (
        classify_context_evidence(
            occurrences=60,
            trades=35,
            exp_lcb=0.1,
            positive_windows_ratio=0.60,
        )
        == "ROBUST_CONTEXT_EDGE"
    )
    assert (
        classify_context_evidence(
            occurrences=40,
            trades=20,
            exp_lcb=0.05,
            positive_windows_ratio=0.50,
        )
        == "WEAK_CONTEXT_EDGE"
    )
    assert (
        classify_context_evidence(
            occurrences=20,
            trades=10,
            exp_lcb=-0.01,
            positive_windows_ratio=0.40,
        )
        == "NOISE"
    )


def test_stage27_9_context_evidence_aggregation() -> None:
    rows = []
    for idx in range(10):
        rows.append(
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "context": "TREND",
                "rulelet": "TrendPullback",
                "context_occurrences": 80,
                "trade_count": 45,
                "exp_lcb": 0.2 if idx < 6 else -0.05,
            }
        )
    for idx in range(8):
        rows.append(
            {
                "symbol": "ETH/USDT",
                "timeframe": "1h",
                "context": "RANGE",
                "rulelet": "RangeFade",
                "context_occurrences": 40,
                "trade_count": 20,
                "exp_lcb": 0.1 if idx < 5 else -0.1,
            }
        )
    summary = evaluate_context_evidence(pd.DataFrame(rows))
    counts = dict(summary.get("counts", {}))
    assert int(counts.get("ROBUST_CONTEXT_EDGE", 0)) == 1
    assert int(counts.get("WEAK_CONTEXT_EDGE", 0)) == 1
    top = list(summary.get("top_edges", []))
    assert top
