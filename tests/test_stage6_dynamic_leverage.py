"""Stage-6 dynamic leverage tests."""

from __future__ import annotations

from buffmini.regime.classifier import REGIME_RANGE, REGIME_TREND, REGIME_VOL_EXPANSION
from buffmini.risk.dynamic_leverage import compute_dynamic_leverage, compute_recent_drawdown


def _cfg() -> dict:
    return {
        "trend_multiplier": 1.2,
        "range_multiplier": 0.9,
        "vol_expansion_multiplier": 0.7,
        "dd_soft_threshold": 0.08,
        "dd_soft_multiplier": 0.8,
        "dd_lookback_bars": 168,
        "max_leverage": 25.0,
        "allowed_levels": [1, 2, 3, 5, 10, 15, 20, 25, 50],
    }


def test_trend_leverage_higher_than_vol_expansion() -> None:
    cfg = _cfg()
    trend = compute_dynamic_leverage(base_leverage=10.0, regime=REGIME_TREND, dd_recent=0.01, config=cfg)
    vol = compute_dynamic_leverage(base_leverage=10.0, regime=REGIME_VOL_EXPANSION, dd_recent=0.01, config=cfg)
    assert float(trend["leverage_clipped"]) > float(vol["leverage_clipped"])


def test_leverage_never_exceeds_max_leverage() -> None:
    cfg = _cfg()
    result = compute_dynamic_leverage(base_leverage=30.0, regime=REGIME_TREND, dd_recent=0.0, config=cfg)
    assert float(result["leverage_clipped"]) <= 25.0


def test_drawdown_soft_threshold_reduces_leverage() -> None:
    cfg = _cfg()
    low_dd = compute_dynamic_leverage(base_leverage=10.0, regime=REGIME_RANGE, dd_recent=0.01, config=cfg)
    high_dd = compute_dynamic_leverage(base_leverage=10.0, regime=REGIME_RANGE, dd_recent=0.20, config=cfg)
    assert float(high_dd["leverage_raw"]) < float(low_dd["leverage_raw"])


def test_recent_drawdown_is_deterministic() -> None:
    equity = [100.0, 105.0, 98.0, 103.0, 95.0, 99.0]
    first = compute_recent_drawdown(equity_history=equity, lookback_bars=4)
    second = compute_recent_drawdown(equity_history=equity, lookback_bars=4)
    assert first == second

