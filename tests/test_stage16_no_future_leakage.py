from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.alpha_v2.context import compute_context_states
from buffmini.data.features import calculate_features
from buffmini.validation.leakage_harness import synthetic_ohlcv


def test_stage16_context_features_no_future_leakage_future_shock() -> None:
    rows = 900
    shock_index = 700
    warmup = 260
    base = synthetic_ohlcv(rows=rows, seed=7)
    shocked = base.copy()
    shocked.loc[shock_index:, "close"] = shocked.loc[shock_index:, "close"] * 5.0
    shocked.loc[shock_index:, "high"] = shocked.loc[shock_index:, "high"] * 5.0
    shocked.loc[shock_index:, "low"] = shocked.loc[shock_index:, "low"] * 5.0
    base_f = calculate_features(base, config={"data": {"include_futures_extras": False}})
    shock_f = calculate_features(shocked, config={"data": {"include_futures_extras": False}})
    a = compute_context_states(base_f)
    b = compute_context_states(shock_f)
    safe_end = max(0, shock_index - warmup)
    cols = [
        "ctx_score_trend",
        "ctx_score_range",
        "ctx_score_vol_expansion",
        "ctx_score_vol_compression",
        "ctx_score_chop",
    ]
    for col in cols:
        x = pd.to_numeric(a[col], errors="coerce").iloc[:safe_end].to_numpy(dtype=float)
        y = pd.to_numeric(b[col], errors="coerce").iloc[:safe_end].to_numpy(dtype=float)
        assert np.allclose(x, y, atol=1e-12, equal_nan=True)

