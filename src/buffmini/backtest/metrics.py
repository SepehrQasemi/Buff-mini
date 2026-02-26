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
            "max_drawdown": max_dd,
            "trade_count": 0.0,
        }

    pnl = trades["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = float((pnl > 0).mean())

    gross_win = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0
    if gross_loss == 0:
        profit_factor = math.inf if gross_win > 0 else 0.0
    else:
        profit_factor = gross_win / abs(gross_loss)

    return {
        "win_rate": win_rate,
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "expectancy": float(pnl.mean()),
        "profit_factor": float(profit_factor),
        "max_drawdown": _max_drawdown(equity_curve),
        "trade_count": float(len(trades)),
    }


def _max_drawdown(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0

    equity = equity_curve["equity"].astype(float)
    peaks = equity.cummax()
    drawdown = (equity - peaks) / peaks.replace(0, pd.NA)
    return abs(float(drawdown.min(skipna=True))) if not drawdown.empty else 0.0
