"""Stage-10.1 regime feature leakage and determinism checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.data.features import calculate_features
from buffmini.stage10.regimes import get_family_score
from buffmini.validation.leakage_harness import (
    apply_future_shock,
    compare_series_no_future_leakage,
    synthetic_ohlcv,
)


def test_stage10_regime_columns_no_future_leakage() -> None:
    frame = synthetic_ohlcv(rows=1000, seed=42)
    shocked = apply_future_shock(frame, shock_index=760)

    base = calculate_features(frame)
    future = calculate_features(shocked)

    checked_columns = [
        "bb_bandwidth_20",
        "bb_bandwidth_z_120",
        "atr_pct",
        "atr_pct_rank_252",
        "ema_slope_50",
        "trend_strength_stage10",
        "trend_strength_rank_252",
        "ema_slope_flip_rate_48",
        "volume_z_120",
        "score_trend",
        "score_range",
        "score_vol_expansion",
        "score_vol_compression",
        "score_chop",
        "regime_confidence_stage10",
    ]
    safe_end = 760 - 280 - 1
    assert safe_end > 0

    for column in checked_columns:
        leaked, _, checked = compare_series_no_future_leakage(
            baseline=base[column],
            shocked=future[column],
            safe_end=safe_end,
            tol=1e-12,
        )
        assert checked > 0
        assert not leaked, f"leak detected in {column}"


def test_stage10_regime_deterministic_labels() -> None:
    frame = synthetic_ohlcv(rows=720, seed=11)
    left = calculate_features(frame)
    right = calculate_features(frame)

    assert left["regime_label_stage10"].astype(str).equals(right["regime_label_stage10"].astype(str))
    assert np.allclose(
        pd.to_numeric(left["regime_confidence_stage10"], errors="coerce").fillna(0.0).to_numpy(),
        pd.to_numeric(right["regime_confidence_stage10"], errors="coerce").fillna(0.0).to_numpy(),
        atol=0.0,
        rtol=0.0,
    )


def test_stage10_family_score_helper_deterministic() -> None:
    frame = synthetic_ohlcv(rows=720, seed=21)
    features = calculate_features(frame)
    row = features.iloc[350]
    left = get_family_score("MA_SlopePullback", row)
    right = get_family_score("MA_SlopePullback", row)
    assert left == right
    assert 0.0 <= left <= 1.0
