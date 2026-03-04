from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage28.usability import UsabilityConfig, compute_usability, pool_returns_for_mc


def test_stage28_mc_pooling_for_rare_contexts() -> None:
    candidate = {
        "context_occurrences": 140,
        "trades_in_context": 8,
    }
    windows = pd.DataFrame(
        {
            "window_id": [0, 1, 2, 3, 4, 5],
            "trade_count": [6, 6, 6, 6, 6, 6],
            "occurrences": [22, 21, 23, 25, 24, 25],
        }
    )
    out = compute_usability(
        candidate=candidate,
        windows=windows,
        cfg=UsabilityConfig(min_trades_context=30, min_occurrences_context=50, min_windows=3, rare_pool_min_trades=30),
    )
    assert bool(out["usable"]) is True
    assert bool(out["mc_pooling"]) is True
    assert bool(out["wf_triggered"]) is True
    assert bool(out["mc_triggered"]) is True
    assert str(out["reason"]) == "rare_context_pooled"


def test_stage28_pool_returns_helper_contract() -> None:
    pooled, ok = pool_returns_for_mc(
        windows_returns=[
            np.asarray([1.0, -0.5, 0.2]),
            np.asarray([0.3, 0.4, -0.1]),
            np.asarray([np.nan, 0.5]),
        ],
        min_total_trades=7,
    )
    assert bool(ok) is True
    assert int(pooled.size) == 7

