"""Stage-10.4 soft activation tests."""

from __future__ import annotations

import numpy as np

from buffmini.data.features import calculate_features
from buffmini.stage10.activation import apply_soft_activation
from buffmini.stage10.signals import generate_signal_family
from buffmini.validation.leakage_harness import synthetic_ohlcv


def _entry_count(signal_values: np.ndarray) -> int:
    prev = np.roll(signal_values, 1)
    prev[0] = 0
    entered = (signal_values != 0) & (prev == 0)
    return int(entered.sum())


def test_soft_activation_keeps_trade_triggers_unchanged() -> None:
    frame = synthetic_ohlcv(rows=900, seed=123)
    features = calculate_features(frame)
    signal_frame = generate_signal_family(features, family="BollingerSnapBack")
    activated = apply_soft_activation(signal_frame, features, signal_family="BollingerSnapBack")

    assert signal_frame["signal"].equals(activated["signal"])
    before_count = _entry_count(signal_frame["signal"].to_numpy(dtype=int))
    after_count = _entry_count(activated["signal"].to_numpy(dtype=int))
    delta_pct = 0.0 if before_count == 0 else abs(after_count - before_count) / float(before_count) * 100.0
    assert delta_pct <= 1.0


def test_activation_multiplier_bounds_and_determinism() -> None:
    frame = synthetic_ohlcv(rows=800, seed=77)
    features = calculate_features(frame)
    signal_frame = generate_signal_family(features, family="MA_SlopePullback")

    left = apply_soft_activation(signal_frame, features, signal_family="MA_SlopePullback")
    right = apply_soft_activation(signal_frame, features, signal_family="MA_SlopePullback")

    mult = left["activation_multiplier"].to_numpy(dtype=float)
    assert np.isfinite(mult).all()
    assert (mult >= 0.9).all()
    assert (mult <= 1.1).all()
    assert left.equals(right)
