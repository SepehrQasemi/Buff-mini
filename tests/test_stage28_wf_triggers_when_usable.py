from __future__ import annotations

import pandas as pd

from buffmini.stage28.usability import UsabilityConfig, compute_usability


def test_stage28_wf_triggers_when_usable() -> None:
    candidate = {
        "context_occurrences": 120,
        "trades_in_context": 36,
    }
    windows = pd.DataFrame(
        {
            "window_id": [0, 1, 2, 3],
            "trade_count": [8, 9, 10, 9],
            "occurrences": [30, 28, 32, 30],
        }
    )
    out = compute_usability(candidate=candidate, windows=windows, cfg=UsabilityConfig())
    assert bool(out["usable"]) is True
    assert bool(out["wf_triggered"]) is True
    assert bool(out["mc_triggered"]) is True
    assert bool(out["mc_pooling"]) is False
    assert str(out["reason"]) == "direct_context_usable"

