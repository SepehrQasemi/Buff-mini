from __future__ import annotations

import numpy as np

from buffmini.alpha_v2.context import STATES, compute_context_states, context_persistence_summary
from buffmini.data.features import calculate_features
from buffmini.validation.leakage_harness import synthetic_ohlcv


def test_stage16_context_is_deterministic_and_transition_rows_sum_to_one() -> None:
    raw = synthetic_ohlcv(rows=1200, seed=42)
    frame = calculate_features(raw, config={"data": {"include_futures_extras": False}})
    a = compute_context_states(frame)
    b = compute_context_states(frame)
    assert (a["ctx_state"].astype(str).to_numpy() == b["ctx_state"].astype(str).to_numpy()).all()
    durations, trans = context_persistence_summary(a)
    assert set(durations["state"].astype(str)) == set(STATES)
    row_sums = trans.sum(axis=1).to_numpy(dtype=float)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-9)

