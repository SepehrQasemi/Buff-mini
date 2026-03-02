from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.alpha_v2.transitions import combined_transition_score, transition_scores
from buffmini.data.features import calculate_features
from buffmini.validation.leakage_harness import synthetic_ohlcv


def _frame(rows: int = 800, seed: int = 42) -> pd.DataFrame:
    base = synthetic_ohlcv(rows=rows, seed=seed)
    return calculate_features(base, config={"data": {"include_futures_extras": False}})


def test_stage19_detects_compression_to_expansion_fixture() -> None:
    frame = _frame(200, 5)
    frame["bb_bandwidth_z_120"] = -1.2
    frame.loc[120:, "bb_bandwidth_z_120"] = 1.2
    scores = transition_scores(frame)
    hit_count = int((pd.to_numeric(scores["tr_cmp_to_exp_breakout"], errors="coerce").abs() > 0).sum())
    assert hit_count > 0


def test_stage19_transition_scores_no_future_leakage() -> None:
    rows = 850
    shock_idx = 650
    warmup = 260
    a = _frame(rows, 11)
    b = a.copy()
    b.loc[shock_idx:, "close"] = b.loc[shock_idx:, "close"] * 4.0
    b.loc[shock_idx:, "high"] = b.loc[shock_idx:, "high"] * 4.0
    b.loc[shock_idx:, "low"] = b.loc[shock_idx:, "low"] * 4.0
    sa = transition_scores(a)
    sb = transition_scores(b)
    safe_end = max(0, shock_idx - warmup)
    for col in sa.columns:
        x = pd.to_numeric(sa[col], errors="coerce").iloc[:safe_end].to_numpy(dtype=float)
        y = pd.to_numeric(sb[col], errors="coerce").iloc[:safe_end].to_numpy(dtype=float)
        assert np.allclose(x, y, atol=1e-12, equal_nan=True)


def test_stage19_transition_frequency_bounded() -> None:
    frame = _frame(1000, 7)
    score = combined_transition_score(frame)
    active_ratio = float((pd.to_numeric(score, errors="coerce").abs() > 0.2).mean())
    assert active_ratio <= 0.5

