from __future__ import annotations

import pandas as pd

from buffmini.stage26.rulelets import build_rulelet_library
from buffmini.stage28.context_discovery import ContextCandidate, compute_context_signal


def _fixture_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="1h", tz="UTC"),
            "open": [100.0, 100.2, 100.4, 100.3, 100.5, 100.7, 100.6, 100.8, 100.9, 101.0, 101.2, 101.1],
            "high": [100.4, 100.5, 100.8, 100.7, 100.9, 101.0, 100.9, 101.1, 101.2, 101.3, 101.4, 101.3],
            "low": [99.8, 100.0, 100.1, 100.0, 100.2, 100.4, 100.3, 100.5, 100.6, 100.7, 100.8, 100.7],
            "close": [100.2, 100.4, 100.5, 100.4, 100.7, 100.8, 100.7, 101.0, 101.1, 101.2, 101.3, 101.2],
            "volume": [1000, 1100, 1200, 1300, 1250, 1280, 1270, 1290, 1310, 1320, 1330, 1340],
            "ctx_state": ["TREND"] * 6 + ["RANGE"] * 6,
        }
    )


def test_stage28_context_signal_only_active_inside_context() -> None:
    frame = _fixture_frame()
    lib = build_rulelet_library()
    candidate = ContextCandidate(
        name="TrendPullback",
        family="price",
        context="TREND",
        threshold=0.0,
        default_exit="fixed_atr",
        required_features=tuple(lib["TrendPullback"].required_features()),
    )
    _, signal, mask = compute_context_signal(
        frame=frame,
        candidate=candidate,
        rulelet_library=lib,
        shift_entries=False,
    )
    nonzero_idx = signal[signal != 0].index
    assert all(bool(mask.iloc[idx]) for idx in nonzero_idx)

