from __future__ import annotations

import pandas as pd

from buffmini.mtf.align import assert_causal_alignment, join_mtf_layer
from buffmini.mtf.spec import MtfLayerSpec


def test_join_mtf_layer_is_backward_causal_and_future_shock_safe() -> None:
    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=10, freq="1h", tz="UTC"),
            "close": range(10),
        }
    )
    layer = pd.DataFrame(
        {
            "ts_open": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z"], utc=True),
            "ts_close": pd.to_datetime(["2026-01-01T04:00:00Z", "2026-01-01T08:00:00Z"], utc=True),
            "trend_strength": [10.0, 20.0],
        }
    )
    spec = MtfLayerSpec(
        name="htf_4h",
        timeframe="4h",
        role="context",
        features=("trend_strength",),
        tolerance_bars=2,
        enabled=True,
    )

    joined_left, _ = join_mtf_layer(base_df=base, layer_df=layer, layer_spec=spec)
    assert_causal_alignment(joined_left, base_ts_col="timestamp", layer_close_col="htf_4h__ts_close")

    shocked = layer.copy()
    shocked.loc[shocked["ts_close"] == pd.Timestamp("2026-01-01T08:00:00Z"), "trend_strength"] = 9999.0
    joined_right, _ = join_mtf_layer(base_df=base, layer_df=shocked, layer_spec=spec)

    mask_before_future_close = pd.to_datetime(joined_left["timestamp"], utc=True) < pd.Timestamp("2026-01-01T08:00:00Z")
    left_vals = joined_left.loc[mask_before_future_close, "htf_4h__trend_strength"].reset_index(drop=True)
    right_vals = joined_right.loc[mask_before_future_close, "htf_4h__trend_strength"].reset_index(drop=True)
    pd.testing.assert_series_equal(left_vals, right_vals, check_names=False)

    matched_mask = joined_right["htf_4h__ts_close"].notna()
    assert (
        pd.to_datetime(joined_right.loc[matched_mask, "htf_4h__ts_close"], utc=True)
        <= pd.to_datetime(joined_right.loc[matched_mask, "timestamp"], utc=True)
    ).all()
