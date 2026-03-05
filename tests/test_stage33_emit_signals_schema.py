from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage33.emitter import emit_signal_payload


def _frame(n: int = 300) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    x = np.linspace(0.0, 12.0, n)
    close = 100.0 + np.sin(x) * 1.5 + x * 0.1
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1000.0 + np.cos(x) * 50.0,
            "ema_50": pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy(),
            "atr_pct": np.full(n, 0.01),
            "ctx_state": ["TREND"] * n,
            "context_prob_0": np.full(n, 0.8),
            "context_prob_1": np.full(n, 0.2),
        }
    )


def test_stage33_emit_signals_schema() -> None:
    frame = _frame()
    policy = {
        "policy_id": "stage33_test",
        "contexts": {
            "TREND": {
                "status": "OK",
                "candidates": [
                    {"candidate_id": "cand_1", "weight": 0.7},
                    {"candidate_id": "cand_2", "weight": 0.3},
                ],
            }
        },
    }
    payload = emit_signal_payload(
        frame=frame,
        policy=policy,
        symbol="BTC/USDT",
        timeframe="1h",
        equity=1000.0,
    )
    required = {
        "symbol",
        "timeframe",
        "asof_ts",
        "context_probabilities",
        "recommended_action",
        "confidence",
        "entry_conditions_summary",
        "stop_exit_policy",
        "sizing",
        "feasibility_notes",
        "explanation",
    }
    assert required.issubset(set(payload.keys()))
    assert payload["recommended_action"] in {"LONG", "SHORT", "FLAT"}
    assert 0.0 <= float(payload["confidence"]) <= 1.0

