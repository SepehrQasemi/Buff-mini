from __future__ import annotations

import pandas as pd

from buffmini.stage26.context import classify_context
from buffmini.validation.leakage_harness import (
    apply_future_shock,
    compare_series_no_future_leakage,
    synthetic_ohlcv,
)


def test_stage26_context_features_and_scores_no_future_leakage() -> None:
    frame = synthetic_ohlcv(rows=1400, seed=4242)
    shocked = apply_future_shock(frame, shock_index=1100)
    base = classify_context(frame)
    future = classify_context(shocked)
    safe_end = 1100 - 320 - 1
    assert safe_end > 0

    checked_cols = [
        "ctx_atr_pct",
        "ctx_realized_vol",
        "ctx_bb_width",
        "ctx_trend_strength",
        "ctx_chop_score",
        "ctx_volume_z",
        "ctx_score_trend",
        "ctx_score_range",
        "ctx_score_vol_expansion",
        "ctx_score_vol_compression",
        "ctx_score_chop",
        "ctx_score_volume_shock",
        "ctx_confidence",
    ]
    for col in checked_cols:
        leaked, _, checked = compare_series_no_future_leakage(
            baseline=base[col],
            shocked=future[col],
            safe_end=safe_end,
            tol=1e-12,
        )
        assert checked > 0
        assert not leaked, f"leak detected in {col}"

    assert base.loc[:safe_end, "ctx_state"].astype(str).tolist() == future.loc[:safe_end, "ctx_state"].astype(str).tolist()


def test_stage26_context_deterministic_labels() -> None:
    frame = synthetic_ohlcv(rows=900, seed=7)
    first = classify_context(frame)
    second = classify_context(frame)

    score_cols = [c for c in first.columns if c.startswith("ctx_score_")] + ["ctx_confidence"]
    for col in score_cols:
        pd.testing.assert_series_equal(first[col], second[col], check_names=False)
    assert first["ctx_state"].astype(str).tolist() == second["ctx_state"].astype(str).tolist()
