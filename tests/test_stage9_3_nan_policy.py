"""Stage-9.3 NaN-safe rule evaluation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.discovery.dsl import (
    FUNDING_NEUTRAL,
    OI_NEUTRAL,
    funding_regime,
    nan_safe_gt,
    oi_regime,
    select_signals_by_regime,
)


def test_nan_safe_gt_returns_false_for_nan_operands() -> None:
    left = pd.Series([1.0, np.nan, 2.0, np.nan])
    right = pd.Series([0.0, 0.0, np.nan, np.nan])
    out = nan_safe_gt(left, right)
    assert out.tolist() == [True, False, False, False]


def test_regime_functions_handle_nan_without_exceptions() -> None:
    frame = pd.DataFrame(
        {
            "funding_extreme_pos": [1.0, np.nan, 0.0],
            "funding_extreme_neg": [0.0, np.nan, 1.0],
            "oi_chg_24": [np.nan, 1.0, -1.0],
        }
    )
    f = funding_regime(frame)
    o = oi_regime(frame)
    assert f.iloc[1] == FUNDING_NEUTRAL
    assert o.iloc[0] == OI_NEUTRAL


def test_select_signals_by_regime_nan_policy_condition_false() -> None:
    frame = pd.DataFrame(
        {
            "funding_extreme_pos": [np.nan, np.nan, np.nan],
            "funding_extreme_neg": [np.nan, np.nan, np.nan],
            "oi_chg_24": [np.nan, np.nan, np.nan],
        }
    )
    primary = pd.Series([1, -1, 1])
    alternate = pd.Series([-1, 1, -1])
    selected = select_signals_by_regime(frame=frame, primary_signal=primary, alternate_signal=alternate)
    assert selected.tolist() == primary.tolist()

