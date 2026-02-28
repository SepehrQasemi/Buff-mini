"""Stage-10.6 activation must not filter entries."""

from __future__ import annotations

import numpy as np

from buffmini.backtest.engine import run_backtest
from buffmini.data.features import calculate_features
from buffmini.stage10.activation import apply_soft_activation
from buffmini.stage10.signals import generate_signal_family
from buffmini.validation.leakage_harness import synthetic_ohlcv


def test_activation_does_not_change_entry_timestamps_or_trade_count() -> None:
    frame = synthetic_ohlcv(rows=1200, seed=1234)
    features = calculate_features(frame)
    signal_frame = generate_signal_family(features, family="ATR_DistanceRevert")

    activation_off = apply_soft_activation(
        signal_frame=signal_frame,
        regime_frame=features,
        signal_family="ATR_DistanceRevert",
        settings={"multiplier_min": 1.0, "multiplier_max": 1.0},
    )
    activation_on = apply_soft_activation(
        signal_frame=signal_frame,
        regime_frame=features,
        signal_family="ATR_DistanceRevert",
    )

    assert activation_off["signal"].equals(activation_on["signal"])
    assert activation_off["long_entry"].equals(activation_on["long_entry"])
    assert activation_off["short_entry"].equals(activation_on["short_entry"])

    work = features.copy()
    work["signal"] = signal_frame["signal"].astype(int)
    baseline = run_backtest(frame=work, strategy_name="s10_6_base", symbol="BTC/USDT")
    with_activation = run_backtest(frame=work, strategy_name="s10_6_act", symbol="BTC/USDT")
    baseline_count = int(baseline.metrics.get("trade_count", 0))
    active_count = int(with_activation.metrics.get("trade_count", 0))
    delta_pct = 0.0 if baseline_count == 0 else abs(active_count - baseline_count) / baseline_count * 100.0
    assert delta_pct <= 0.1

    multipliers = activation_on["activation_multiplier"].to_numpy(dtype=float)
    assert np.isfinite(multipliers).all()
    assert (multipliers >= 0.9).all()
    assert (multipliers <= 1.1).all()
