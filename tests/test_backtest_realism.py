from __future__ import annotations

import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.backtest.metrics import calculate_metrics


def _hold_frame(hours: int = 72) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    close = [100.0 + (i * 0.02) for i in range(hours)]
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": [c + 0.2 for c in close],
            "low": [c - 0.2 for c in close],
            "close": close,
            "volume": [1000.0] * hours,
            "atr_14": [1.0] * hours,
            "signal": [1] + [0] * (hours - 1),
        }
    )


def test_funding_cost_reduces_equity() -> None:
    frame = _hold_frame(72)
    no_funding = run_backtest(
        frame=frame,
        strategy_name="t",
        symbol="BTC/USDT",
        max_hold_bars=48,
        round_trip_cost_pct=0.0,
        slippage_pct=0.0,
        funding_pct_per_day=0.0,
    )
    with_funding = run_backtest(
        frame=frame,
        strategy_name="t",
        symbol="BTC/USDT",
        max_hold_bars=48,
        round_trip_cost_pct=0.0,
        slippage_pct=0.0,
        funding_pct_per_day=0.001,
    )
    assert float(with_funding.equity_curve["equity"].iloc[-1]) < float(no_funding.equity_curve["equity"].iloc[-1])


def test_risk_budget_position_sizing_is_more_conservative_than_full_equity() -> None:
    frame = _hold_frame(36)
    full = run_backtest(
        frame=frame,
        strategy_name="t",
        symbol="BTC/USDT",
        max_hold_bars=24,
        round_trip_cost_pct=0.0,
        slippage_pct=0.0,
        position_sizing_mode="full_equity",
    )
    risk = run_backtest(
        frame=frame,
        strategy_name="t",
        symbol="BTC/USDT",
        max_hold_bars=24,
        round_trip_cost_pct=0.0,
        slippage_pct=0.0,
        position_sizing_mode="risk_budget",
        risk_per_trade_pct=0.005,
    )
    assert float(risk.trades["entry_price"].iloc[0]) == float(full.trades["entry_price"].iloc[0])
    assert abs(float(risk.trades["pnl"].iloc[0])) <= abs(float(full.trades["pnl"].iloc[0]))


def test_metrics_sanitize_infinite_profit_factor() -> None:
    trades = pd.DataFrame(
        [
            {"pnl": 10.0},
            {"pnl": 5.0},
            {"pnl": 2.0},
        ]
    )
    equity = pd.DataFrame({"equity": [10000.0, 10010.0, 10015.0, 10017.0]})
    metrics = calculate_metrics(trades, equity)
    assert metrics["profit_factor"] == 10.0
    assert metrics["metrics_sanitized"] == 1.0
