from __future__ import annotations

import pandas as pd

from buffmini.stage26.policy import build_conditional_policy, compose_policy_signal


def _effects() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rulelet": "TrendPullback",
                "family": "price",
                "context": "TREND",
                "context_occurrences": 180,
                "trades_in_context": 90,
                "expectancy": 0.8,
                "exp_lcb": 0.5,
                "classification": "PASS",
            },
            {
                "rulelet": "StructureBreak",
                "family": "price",
                "context": "TREND",
                "context_occurrences": 180,
                "trades_in_context": 70,
                "expectancy": 0.5,
                "exp_lcb": 0.3,
                "classification": "PASS",
            },
            {
                "rulelet": "RangeFade",
                "family": "price",
                "context": "RANGE",
                "context_occurrences": 150,
                "trades_in_context": 55,
                "expectancy": 0.4,
                "exp_lcb": 0.2,
                "classification": "PASS",
            },
            {
                "rulelet": "Noise",
                "family": "price",
                "context": "RANGE",
                "context_occurrences": 150,
                "trades_in_context": 5,
                "expectancy": -0.2,
                "exp_lcb": -0.1,
                "classification": "FAIL",
            },
        ]
    )


def test_stage26_policy_builds_context_weights() -> None:
    policy = build_conditional_policy(
        effects=_effects(),
        min_occurrences_per_context=30,
        min_trades_in_context=20,
        top_k=2,
        w_min=0.05,
        w_max=0.80,
    )
    assert "contexts" in policy
    trend = dict(policy["contexts"]["TREND"])
    assert trend["status"] == "OK"
    assert len(trend["rulelets"]) == 2
    w_sum = sum(float(v) for v in trend["weights"].values())
    assert abs(w_sum - 1.0) < 1e-9
    for value in trend["weights"].values():
        assert 0.0 <= float(value) <= 1.0

    rng = dict(policy["contexts"]["RANGE"])
    assert rng["status"] == "OK"
    assert "Noise" not in rng["rulelets"]


def test_stage26_policy_compose_signal_conflict_modes() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01T00:00:00Z", periods=5, freq="1h", tz="UTC"),
            "ctx_state": ["TREND", "TREND", "RANGE", "RANGE", "TREND"],
        }
    )
    policy = {
        "contexts": {
            "TREND": {"weights": {"A": 0.6, "B": 0.4}},
            "RANGE": {"weights": {"A": 0.2, "B": 0.8}},
        }
    }
    rulelet_scores = {
        "A": pd.Series([0.8, 0.7, -0.4, -0.2, 0.5], index=frame.index, dtype=float),
        "B": pd.Series([-0.3, 0.2, -0.7, 0.5, 0.4], index=frame.index, dtype=float),
    }
    net_signal, net_trace = compose_policy_signal(
        frame=frame,
        rulelet_scores=rulelet_scores,
        policy=policy,
        conflict_mode="net",
    )
    hedge_signal, hedge_trace = compose_policy_signal(
        frame=frame,
        rulelet_scores=rulelet_scores,
        policy=policy,
        conflict_mode="hedge",
    )
    assert len(net_signal) == len(frame)
    assert len(hedge_signal) == len(frame)
    assert list(net_trace["conflict_mode"].unique()) == ["net"]
    assert list(hedge_trace["conflict_mode"].unique()) == ["hedge"]
    # Shifted output starts neutral then follows composed direction.
    assert int(net_signal.iloc[0]) == 0
    assert set(net_signal.astype(int).unique()).issubset({-1, 0, 1})
