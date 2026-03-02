from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.alpha_v2.conditional_tests import (
    ConditionalTestConfig,
    apply_falsification_rules,
    conditional_effects_table,
)


def test_stage18_detects_known_conditional_edge() -> None:
    rng = np.random.default_rng(42)
    n = 1200
    context = np.where(np.arange(n) % 2 == 0, "TREND", "RANGE")
    signal = rng.integers(0, 2, size=n)
    noise = rng.normal(0.0, 0.2, n)
    fwd = noise + np.where((context == "TREND") & (signal == 1), 0.15, 0.0)
    frame = pd.DataFrame({"ctx": context, "sig": signal, "fwd": fwd})
    table = conditional_effects_table(
        frame=frame,
        signal_col="sig",
        context_col="ctx",
        forward_return_col="fwd",
        cfg=ConditionalTestConfig(bootstrap_samples=600, seed=42, min_samples=30),
    )
    table = apply_falsification_rules(table=table, min_samples=30)
    trend = table.loc[table["context"] == "TREND"].iloc[0]
    assert bool(trend["accepted"])
    assert float(trend["median_diff"]) > 0.0


def test_stage18_rejects_random_signal() -> None:
    rng = np.random.default_rng(7)
    n = 1400
    frame = pd.DataFrame(
        {
            "ctx": np.where(np.arange(n) % 3 == 0, "TREND", "RANGE"),
            "sig": rng.integers(0, 2, size=n),
            "fwd": rng.normal(0.0, 0.2, n),
        }
    )
    table = conditional_effects_table(
        frame=frame,
        signal_col="sig",
        context_col="ctx",
        forward_return_col="fwd",
        cfg=ConditionalTestConfig(bootstrap_samples=500, seed=7, min_samples=30),
    )
    table = apply_falsification_rules(table=table, min_samples=30)
    assert not bool(table["accepted"].any())
