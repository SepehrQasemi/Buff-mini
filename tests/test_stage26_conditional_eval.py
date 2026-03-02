from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from buffmini.stage26.conditional_eval import ConditionalEvalParams, evaluate_rulelets_conditionally


@dataclass(frozen=True)
class _DummyRulelet:
    name: str
    family: str
    contexts_allowed: tuple[str, ...]
    threshold: float
    default_exit: str
    score: pd.Series

    def compute_score(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.score.to_numpy(dtype=float), index=df.index, dtype=float)


def _frame(rows: int = 1200) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="1h", tz="UTC")
    trend = np.linspace(100.0, 160.0, num=rows, dtype=float)
    wiggle = np.sin(np.linspace(0.0, 30.0, num=rows, dtype=float)) * 0.15
    close = trend + wiggle
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 0.4
    low = np.minimum(open_, close) - 0.4
    volume = np.full(rows, 20.0, dtype=float)
    atr = np.full(rows, 0.8, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr_14": atr,
            "ctx_state": ["TREND"] * rows,
        }
    )


def test_stage26_conditional_eval_detects_known_conditional_edge() -> None:
    frame = _frame()
    edge_score = pd.Series(0.9, index=frame.index, dtype=float)
    rulelets = {
        "EdgeRule": _DummyRulelet(
            name="EdgeRule",
            family="price",
            contexts_allowed=("TREND",),
            threshold=0.2,
            default_exit="fixed_atr",
            score=edge_score,
        )
    }
    table, details = evaluate_rulelets_conditionally(
        frame=frame,
        rulelets=rulelets,
        symbol="BTC/USDT",
        timeframe="1h",
        seed=42,
        cost_levels=[
            {"name": "realistic", "round_trip_cost_pct": 0.1, "slippage_pct": 0.0005, "cost_model_cfg": {}},
            {"name": "high", "round_trip_cost_pct": 0.2, "slippage_pct": 0.001, "cost_model_cfg": {}},
        ],
        params=ConditionalEvalParams(
            bootstrap_samples=200,
            seed=42,
            min_occurrences=30,
            min_trades=20,
            rare_min_trades=8,
            rolling_months=(3, 6),
        ),
    )
    assert not table.empty
    row = table.iloc[0].to_dict()
    assert row["rulelet"] == "EdgeRule"
    assert row["context"] == "TREND"
    assert int(row["context_occurrences"]) == len(frame)
    assert int(row["trades_in_context"]) > 0
    assert float(row["exp_lcb"]) > 0.0
    assert str(row["classification"]) in {"PASS", "RARE", "WEAK"}
    assert isinstance(details.get("rows"), list)


def test_stage26_conditional_eval_rejects_noise_rulelet() -> None:
    frame = _frame()
    noise_score = pd.Series(0.0, index=frame.index, dtype=float)
    rulelets = {
        "NoiseRule": _DummyRulelet(
            name="NoiseRule",
            family="price",
            contexts_allowed=("TREND",),
            threshold=0.2,
            default_exit="fixed_atr",
            score=noise_score,
        )
    }
    table, _ = evaluate_rulelets_conditionally(
        frame=frame,
        rulelets=rulelets,
        symbol="BTC/USDT",
        timeframe="1h",
        seed=7,
        cost_levels=[{"name": "realistic", "round_trip_cost_pct": 0.1, "slippage_pct": 0.0005, "cost_model_cfg": {}}],
        params=ConditionalEvalParams(
            bootstrap_samples=100,
            seed=7,
            min_occurrences=20,
            min_trades=10,
            rare_min_trades=5,
            rolling_months=(3,),
        ),
    )
    assert not table.empty
    row = table.iloc[0].to_dict()
    assert int(row["trades_in_context"]) == 0
    assert float(row["exp_lcb"]) <= 0.0
    assert str(row["classification"]) == "FAIL"
