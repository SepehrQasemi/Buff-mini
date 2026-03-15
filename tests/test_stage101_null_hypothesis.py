from __future__ import annotations

import pandas as pd

from buffmini.research.null_hypothesis import (
    candidate_beats_control,
    mean_reversion_baseline_signal,
    momentum_baseline_signal,
    randomized_signal,
)


def test_randomized_signal_is_deterministic_and_preserves_activity() -> None:
    base = pd.Series([0, 1, 0, -1, 0, 1, 0, 0], dtype=int)
    first = randomized_signal(base, seed_key="abc")
    second = randomized_signal(base, seed_key="abc")
    assert first.tolist() == second.tolist()
    assert int((first != 0).sum()) == int((base != 0).sum())


def test_candidate_beats_control_requires_stronger_metrics() -> None:
    assert candidate_beats_control(
        {"exp_lcb": 0.02, "expectancy": 0.01, "profit_factor": 1.3},
        {"exp_lcb": 0.0, "expectancy": 0.005, "profit_factor": 1.1},
    )
    assert not candidate_beats_control(
        {"exp_lcb": 0.0, "expectancy": 0.004, "profit_factor": 1.05},
        {"exp_lcb": 0.01, "expectancy": 0.003, "profit_factor": 1.0},
    )


def test_baseline_signals_match_frame_length() -> None:
    frame = pd.DataFrame({"close": [100 + i for i in range(80)]})
    assert len(momentum_baseline_signal(frame)) == len(frame)
    assert len(mean_reversion_baseline_signal(frame)) == len(frame)
