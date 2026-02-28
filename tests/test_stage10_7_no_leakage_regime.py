"""Stage-10.7 no-leakage checks for calibrated regime features."""

from __future__ import annotations

from buffmini.data.features import calculate_features
from buffmini.validation.leakage_harness import (
    apply_future_shock,
    compare_series_no_future_leakage,
    synthetic_ohlcv,
)


def test_stage10_7_regime_calibration_features_no_future_leakage() -> None:
    frame = synthetic_ohlcv(rows=1200, seed=1337)
    shocked = apply_future_shock(frame, shock_index=900)
    base = calculate_features(frame)
    future = calculate_features(shocked)

    checked_columns = [
        "trend_strength_rank_252",
        "ema_slope_flip_rate_48",
        "score_trend",
        "score_range",
        "score_vol_expansion",
        "score_vol_compression",
        "score_chop",
        "regime_confidence_stage10",
    ]
    safe_end = 900 - 320 - 1
    assert safe_end > 0
    for col in checked_columns:
        leaked, _, checked = compare_series_no_future_leakage(
            baseline=base[col],
            shocked=future[col],
            safe_end=safe_end,
            tol=1e-12,
        )
        assert checked > 0
        assert not leaked, f"leak detected in {col}"
