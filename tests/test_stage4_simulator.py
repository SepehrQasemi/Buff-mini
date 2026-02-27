"""Stage-4 simulator tests."""

from __future__ import annotations

import pandas as pd

from buffmini.execution.policy import Signal
from buffmini.execution.simulator import simulate_execution


def _cfg() -> dict:
    return {
        "execution": {
            "mode": "net",
            "per_symbol_netting": True,
            "allow_opposite_signals": False,
            "symbol_scope": "per_symbol",
        },
        "risk": {
            "max_gross_exposure": 1.0,
            "max_net_exposure_per_symbol": 1.0,
            "max_open_positions": 10,
            "sizing": {
                "mode": "fixed_fraction",
                "risk_per_trade_pct": 1.0,
                "fixed_fraction_pct": 80.0,
            },
            "killswitch": {
                "enabled": True,
                "max_daily_loss_pct": 1.0,
                "max_peak_to_valley_dd_pct": 5.0,
                "max_consecutive_losses": 2,
                "cool_down_bars": 2,
            },
            "reeval": {"cadence": "weekly", "min_new_bars": 168},
        },
        "_method_weights": {"s1": 1.0, "s2": 1.0},
        "_forced_pnl_by_index": {1: -200.0, 2: -50.0},
    }


def test_simulator_outputs_artifacts_and_respects_caps() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    signals = []
    for i in range(6):
        ts = start + pd.Timedelta(hours=i)
        signals.append(Signal(ts=ts, symbol="BTC/USDT", direction=1, strength=1.0, stop_distance=0.02, strategy_id="s1"))
        signals.append(Signal(ts=ts, symbol="ETH/USDT", direction=1, strength=1.0, stop_distance=0.02, strategy_id="s2"))

    metrics, exposure_df, orders_df, killswitch_df = simulate_execution(
        signals_by_ts=signals,
        cfg=_cfg(),
        initial_equity=10000.0,
        chosen_leverage=2.0,
        seed=42,
    )

    assert not orders_df.empty
    assert not exposure_df.empty
    assert float(exposure_df["gross_exposure"].max()) <= 1.0 + 1e-9
    assert int(metrics["scaled_event_count"]) >= 1
    assert int(metrics["killswitch_event_count"]) >= 1
    assert not killswitch_df.empty

