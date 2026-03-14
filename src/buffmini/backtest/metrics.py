"""Backtest metrics."""

from __future__ import annotations

import math

import pandas as pd


def calculate_metrics(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> dict[str, float]:
    """Calculate core trade and equity metrics."""

    if trades.empty:
        max_dd = _max_drawdown(equity_curve)
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "profit_factor_raw": 0.0,
            "max_drawdown": max_dd,
            "trade_count": 0.0,
            "metrics_sanitized": 1.0,
        }

    pnl = trades["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = float((pnl > 0).mean())

    gross_win = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0
    if gross_loss == 0:
        profit_factor_raw = math.inf if gross_win > 0 else 0.0
    else:
        profit_factor_raw = gross_win / abs(gross_loss)
    expectancy = float(pnl.mean())
    profit_factor = _sanitize_profit_factor(profit_factor_raw, expectancy=expectancy)

    return {
        "win_rate": _finite(win_rate),
        "avg_win": _finite(float(wins.mean()) if not wins.empty else 0.0),
        "avg_loss": _finite(float(losses.mean()) if not losses.empty else 0.0),
        "expectancy": _finite(expectancy),
        "profit_factor": _finite(float(profit_factor)),
        "profit_factor_raw": _finite(float(profit_factor_raw), default=0.0),
        "max_drawdown": _finite(_max_drawdown(equity_curve)),
        "trade_count": float(len(trades)),
        "metrics_sanitized": 1.0,
    }


def _max_drawdown(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0

    equity = equity_curve["equity"].astype(float)
    peaks = equity.cummax()
    drawdown = (equity - peaks) / peaks.replace(0, pd.NA)
    return abs(float(drawdown.min(skipna=True))) if not drawdown.empty else 0.0


def _sanitize_profit_factor(value: float, *, expectancy: float) -> float:
    numeric = float(value)
    if math.isnan(numeric):
        return 0.0
    if math.isinf(numeric):
        return 10.0 if float(expectancy) > 0.0 else 0.0
    if numeric < 0:
        return 0.0
    return float(min(numeric, 10.0))


def _finite(value: float, *, default: float = 0.0) -> float:
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return float(default)
    return numeric
