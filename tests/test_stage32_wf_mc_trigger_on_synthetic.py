from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage31.dsl import DSLStrategy
from buffmini.stage32.validate import ValidationConfig, validate_candidates


def _frame(n: int = 2500) -> pd.DataFrame:
    ts = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    x = np.linspace(0.0, 100.0, n)
    close = 100.0 + np.sin(x / 2.0) * 2.0 + x * 0.01
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1000.0 + np.cos(x / 4.0) * 80.0,
            "atr_14": np.maximum(0.5, np.abs(np.cos(x / 8.0)) * 1.2),
        }
    )


def test_stage32_wf_mc_trigger_on_synthetic() -> None:
    frame = _frame()
    strategy = DSLStrategy(
        name="always_long",
        long_expr={"op": ">", "left": {"op": "feature", "name": "close"}, "right": {"op": "const", "value": 0.0}},
        short_expr={"op": "<", "left": {"op": "feature", "name": "close"}, "right": {"op": "const", "value": -999999.0}},
        max_hold_bars=8,
    )
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "cand_long",
                "strategy": strategy,
                "exp_lcb": 0.1,
                "pf_adj": 1.1,
                "maxdd_p95": 0.2,
                "repeatability": 0.6,
                "feasibility_score": 1.0,
            }
        ]
    )
    validated, summary = validate_candidates(
        frame=frame,
        candidates=candidates,
        symbol="BTC/USDT",
        timeframe="1h",
        cfg=ValidationConfig(
            train_days=30,
            holdout_days=10,
            forward_days=10,
            step_days=10,
            min_trades_window=2,
            min_exposure_window=0.01,
            min_usable_windows=2,
            mc_min_trades=5,
            seed=42,
        ),
    )
    assert not validated.empty
    assert float(summary["wf_executed_pct"]) > 0.0
    assert float(summary["mc_trigger_pct"]) > 0.0

