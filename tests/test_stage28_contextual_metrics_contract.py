from __future__ import annotations

import pandas as pd

from buffmini.stage28.context_discovery import ContextCandidate, evaluate_candidate_in_context


def _fixture_frame() -> pd.DataFrame:
    n = 240
    base = pd.Series(range(n), dtype=float)
    close = 100.0 + 0.05 * base + (base % 7) * 0.02
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"),
            "open": close - 0.03,
            "high": close + 0.08,
            "low": close - 0.08,
            "close": close,
            "volume": 1000.0 + (base % 9) * 10.0,
            "ctx_state": ["TREND"] * n,
        }
    )


def test_stage28_contextual_metrics_contract_and_determinism() -> None:
    frame = _fixture_frame()
    candidate = ContextCandidate(
        name="TrendPullback",
        family="price",
        context="TREND",
        threshold=0.2,
        default_exit="fixed_atr",
        required_features=("close",),
    )
    cost_levels = [
        {
            "name": "realistic",
            "round_trip_cost_pct": 0.1,
            "slippage_pct": 0.0005,
            "cost_model_cfg": {},
            "stop_atr_multiple": 1.5,
            "take_profit_atr_multiple": 3.0,
            "max_hold_bars": 24,
        }
    ]
    first = evaluate_candidate_in_context(
        frame=frame,
        candidate=candidate,
        symbol="BTC/USDT",
        timeframe="1h",
        seed=42,
        cost_levels=cost_levels,
    )
    second = evaluate_candidate_in_context(
        frame=frame,
        candidate=candidate,
        symbol="BTC/USDT",
        timeframe="1h",
        seed=42,
        cost_levels=cost_levels,
    )

    for key in (
        "context_occurrences",
        "trades_in_context",
        "expectancy",
        "exp_lcb",
        "profit_factor",
        "max_drawdown",
        "cost_sensitivity",
        "classification",
    ):
        assert key in first
    assert first["context_occurrences"] == second["context_occurrences"]
    assert first["trades_in_context"] == second["trades_in_context"]
    assert float(first["expectancy"]) == float(second["expectancy"])
    assert float(first["exp_lcb"]) == float(second["exp_lcb"])
    assert str(first["classification"]) == str(second["classification"])

