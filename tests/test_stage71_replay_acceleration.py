from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage71 import measure_replay_acceleration


def test_stage71_measures_speedup() -> None:
    candidates = pd.DataFrame([{"candidate_id": f"c{i}"} for i in range(800)])
    returns = np.sin(np.arange(3000, dtype=float) / 13.0) * 0.001
    out = measure_replay_acceleration(
        candidates=candidates,
        returns=returns,
        data_hash="d1",
        setup_signature="s1",
        timeframe="1h",
        cost_model="simple",
        scope_id="scope",
    )
    assert out["status"] == "SUCCESS"
    assert out["baseline_runtime_seconds"] > 0.0
    assert out["optimized_runtime_seconds"] >= 0.0
    assert "cache_key" in out


def test_stage71_is_deterministic_for_same_inputs() -> None:
    candidates = pd.DataFrame([{"candidate_id": f"c{i}"} for i in range(200)])
    returns = np.sin(np.arange(1200, dtype=float) / 11.0) * 0.001
    out1 = measure_replay_acceleration(
        candidates=candidates,
        returns=returns,
        data_hash="d1",
        setup_signature="s1",
        timeframe="1h",
        cost_model="simple",
        scope_id="scope",
    )
    out2 = measure_replay_acceleration(
        candidates=candidates,
        returns=returns,
        data_hash="d1",
        setup_signature="s1",
        timeframe="1h",
        cost_model="simple",
        scope_id="scope",
    )
    assert out1["baseline_total"] == out2["baseline_total"]
    assert out1["optimized_total"] == out2["optimized_total"]
