from __future__ import annotations

import pandas as pd

from buffmini.stage26.conditional_eval import ConditionalEvalParams, evaluate_rulelets_conditionally
from buffmini.stage26.rulelets import build_rulelet_library
from buffmini.utils.hashing import stable_hash


def _frame(rows: int = 480) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="1h", tz="UTC")
    close = pd.Series(range(rows), dtype=float) * 0.2 + 100.0
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": 1000.0,
            "atr_14": 0.9,
            "ctx_state": ["RANGE" if i % 2 == 0 else "TREND" for i in range(rows)],
        }
    )
    return frame


def _hash(df: pd.DataFrame) -> str:
    if df.empty:
        return stable_hash([], length=16)
    keep = [
        "rulelet",
        "family",
        "context",
        "context_occurrences",
        "trades_in_context",
        "expectancy",
        "exp_lcb",
        "max_drawdown",
        "classification",
    ]
    cols = [c for c in keep if c in df.columns]
    view = df.loc[:, cols].sort_values(cols).reset_index(drop=True)
    return stable_hash(view.to_dict(orient="records"), length=16)


def test_stage27_batch_mode_semantically_equivalent() -> None:
    frame = _frame()
    rulelets = build_rulelet_library()
    params = ConditionalEvalParams(
        bootstrap_samples=50,
        seed=42,
        min_occurrences=5,
        min_trades=3,
        rare_min_trades=1,
        rolling_months=(3, 6),
    )
    cost_rows = [
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
    old_df, _ = evaluate_rulelets_conditionally(
        frame=frame,
        rulelets=rulelets,
        symbol="BTC/USDT",
        timeframe="1h",
        seed=42,
        cost_levels=cost_rows,
        params=params,
        batch_mode=False,
    )
    new_df, _ = evaluate_rulelets_conditionally(
        frame=frame,
        rulelets=rulelets,
        symbol="BTC/USDT",
        timeframe="1h",
        seed=42,
        cost_levels=cost_rows,
        params=params,
        batch_mode=True,
    )
    assert _hash(old_df) == _hash(new_df)

