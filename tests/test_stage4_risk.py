"""Stage-4 risk engine tests."""

from __future__ import annotations

import pandas as pd

from buffmini.execution.risk import PortfolioState, compute_position_size, enforce_exposure_caps, killswitch_update_and_decide


def _risk_cfg() -> dict:
    return {
        "max_gross_exposure": 5.0,
        "max_net_exposure_per_symbol": 4.0,
        "max_open_positions": 10,
        "sizing": {
            "mode": "risk_budget",
            "risk_per_trade_pct": 1.0,
            "fixed_fraction_pct": 10.0,
        },
        "killswitch": {
            "enabled": True,
            "max_daily_loss_pct": 5.0,
            "max_peak_to_valley_dd_pct": 20.0,
            "max_consecutive_losses": 8,
            "cool_down_bars": 3,
        },
        "reeval": {"cadence": "weekly", "min_new_bars": 168},
    }


def test_risk_budget_position_size_formula() -> None:
    size = compute_position_size(
        equity=10000.0,
        risk_cfg=_risk_cfg(),
        stop_distance_pct=0.02,
    )
    assert round(size, 6) == 0.5


def test_exposure_caps_scale_and_respect_limits() -> None:
    desired = [
        {"symbol": "BTC/USDT", "exposure_fraction": 1.0},
        {"symbol": "ETH/USDT", "exposure_fraction": 1.0},
    ]
    scaled, multiplier, reasons = enforce_exposure_caps(
        desired_exposures=desired,
        leverage=5.0,
        risk_cfg=_risk_cfg(),
    )

    assert multiplier < 1.0
    assert reasons
    gross = sum(abs(float(item["exposure_fraction"])) * 5.0 for item in scaled)
    assert gross <= 5.0 + 1e-9
    by_symbol = {}
    for item in scaled:
        by_symbol[item["symbol"]] = by_symbol.get(item["symbol"], 0.0) + float(item["exposure_fraction"]) * 5.0
    assert all(abs(value) <= 4.0 + 1e-9 for value in by_symbol.values())


def test_killswitch_triggers_and_cooldown_is_deterministic() -> None:
    cfg = _risk_cfg()
    state = PortfolioState(
        equity=1000.0,
        peak_equity=1000.0,
        day_start_equity=1000.0,
    )
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    first = killswitch_update_and_decide(state, pnl_change=-100.0, ts=ts, bar_index=0, cfg=cfg)
    assert first.allow_new_trades is False
    assert "max_daily_loss_pct" in first.reasons

    second = killswitch_update_and_decide(state, pnl_change=200.0, ts=ts + pd.Timedelta(hours=1), bar_index=1, cfg=cfg)
    third = killswitch_update_and_decide(state, pnl_change=0.0, ts=ts + pd.Timedelta(hours=2), bar_index=2, cfg=cfg)
    fourth = killswitch_update_and_decide(state, pnl_change=0.0, ts=ts + pd.Timedelta(hours=3), bar_index=3, cfg=cfg)
    fifth = killswitch_update_and_decide(state, pnl_change=0.0, ts=ts + pd.Timedelta(hours=4), bar_index=4, cfg=cfg)

    assert second.allow_new_trades is False
    assert third.allow_new_trades is False
    assert fourth.allow_new_trades is False
    assert fifth.allow_new_trades is True
