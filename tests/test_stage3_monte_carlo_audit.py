"""Stage-3.1 Monte Carlo audit-focused tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from buffmini.portfolio.monte_carlo import (
    compute_equity_path_metrics,
    sample_block_indices,
    simulate_equity_paths,
    summarize_mc,
)


def test_mc_determinism_same_seed_identical_summary() -> None:
    pnls = pd.Series([120.0, -60.0, 30.0, -10.0, 45.0, -20.0], dtype=float)
    paths_a = simulate_equity_paths(pnls, n_paths=2000, method="block", seed=42, initial_equity=10000.0, block_size_trades=3)
    paths_b = simulate_equity_paths(pnls, n_paths=2000, method="block", seed=42, initial_equity=10000.0, block_size_trades=3)
    summary_a = summarize_mc(paths_a, initial_equity=10000.0, ruin_dd_threshold=0.5)
    summary_b = summarize_mc(paths_b, initial_equity=10000.0, ruin_dd_threshold=0.5)

    pd.testing.assert_frame_equal(paths_a, paths_b)
    assert summary_a == summary_b


def test_block_bootstrap_contiguous_blocks() -> None:
    rng = np.random.default_rng(42)
    indices = sample_block_indices(n_trades=20, n_paths=5, block_size_trades=4, rng=rng)
    assert indices.shape == (5, 20)
    for row in indices:
        for start in range(0, 20, 4):
            segment = row[start : min(start + 4, 20)]
            if len(segment) > 1:
                diffs = np.diff(segment)
                assert (diffs == 1).all()


def test_max_drawdown_exact_on_known_sequence() -> None:
    metrics = compute_equity_path_metrics([100.0, -50.0, -100.0, 50.0], initial_equity=1000.0)
    expected_max_dd = 150.0 / 1100.0
    assert metrics["max_drawdown"] == expected_max_dd


def test_ruin_probability_in_bounds() -> None:
    pnls = pd.Series([300.0, -900.0, 200.0, -100.0, 50.0], dtype=float)
    paths = simulate_equity_paths(pnls, n_paths=2000, method="iid", seed=7, initial_equity=10000.0)
    summary = summarize_mc(paths, initial_equity=10000.0, ruin_dd_threshold=0.5)
    ruin = float(summary["tail_probabilities"]["p_ruin"])
    assert 0.0 <= ruin <= 1.0


def test_summary_has_no_nan_or_inf() -> None:
    pnls = pd.Series([150.0, -70.0, 40.0, -30.0, 20.0], dtype=float)
    paths = simulate_equity_paths(pnls, n_paths=2000, method="iid", seed=123, initial_equity=10000.0)
    summary = summarize_mc(paths, initial_equity=10000.0, ruin_dd_threshold=0.5)

    def _finite(value: object) -> bool:
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(_finite(item) for item in value.values())
        if isinstance(value, list):
            return all(_finite(item) for item in value)
        return True

    assert _finite(summary)
