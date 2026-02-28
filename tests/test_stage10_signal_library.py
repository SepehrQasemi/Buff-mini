"""Stage-10.2 signal library sanity tests."""

from __future__ import annotations

import numpy as np

from buffmini.data.features import calculate_features
from buffmini.stage10.signals import SIGNAL_FAMILIES, generate_signal_family, resolve_enabled_families
from buffmini.validation.leakage_harness import synthetic_ohlcv


def test_stage10_signal_families_output_contract() -> None:
    frame = synthetic_ohlcv(rows=720, seed=7)
    features = calculate_features(frame)

    for family in SIGNAL_FAMILIES:
        out = generate_signal_family(features, family=family)
        assert len(out) == len(features)
        assert set(out.columns) == {"long_entry", "short_entry", "signal_strength", "signal", "signal_family"}
        assert out["signal"].isin([-1, 0, 1]).all()
        assert np.isfinite(out["signal_strength"].to_numpy(dtype=float)).all()
        assert (out["signal_strength"].to_numpy(dtype=float) >= 0.0).all()
        assert (out["signal_strength"].to_numpy(dtype=float) <= 1.0).all()
        assert (out["signal_family"] == family).all()


def test_stage10_signal_generation_deterministic() -> None:
    frame = synthetic_ohlcv(rows=720, seed=9)
    features = calculate_features(frame)
    left = generate_signal_family(features, family="BollingerSnapBack")
    right = generate_signal_family(features, family="BollingerSnapBack")
    assert left.equals(right)


def test_stage10_enabled_family_resolution_subset() -> None:
    resolved = resolve_enabled_families(
        families=list(SIGNAL_FAMILIES),
        enabled_families=["ATR_DistanceRevert", "unknown", "BreakoutRetest"],
    )
    assert resolved == ["ATR_DistanceRevert", "BreakoutRetest"]
