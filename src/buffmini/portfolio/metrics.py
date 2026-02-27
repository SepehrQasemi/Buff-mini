"""Portfolio-level metric calculations for Stage-2."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


ANNUAL_BARS_1H = 24 * 365
INITIAL_PORTFOLIO_CAPITAL = 10_000.0


def compute_portfolio_metrics(
    returns: pd.Series,
    equity: pd.Series,
    trade_pnls: pd.Series,
    exposure: pd.Series,
) -> dict[str, float | str]:
    """Compute Stage-2 portfolio metrics from bar returns and weighted trades."""

    returns = returns.astype(float).sort_index()
    equity = equity.astype(float).sort_index()
    trade_pnls = trade_pnls.astype(float).reset_index(drop=True)
    exposure = exposure.astype(float).sort_index()

    duration_days = _duration_days(equity.index)
    trade_count_total = float(len(trade_pnls))
    trades_per_month = float(trade_count_total / (duration_days / 30.0)) if duration_days > 0 else 0.0

    expectancy = float(trade_pnls.mean()) if not trade_pnls.empty else 0.0
    trade_std = float(trade_pnls.std(ddof=0)) if not trade_pnls.empty else 0.0
    exp_lcb = float(expectancy - trade_std / math.sqrt(max(1.0, trade_count_total)))

    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    gross_win = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0
    if gross_loss == 0.0:
        profit_factor = math.inf if gross_win > 0 else 0.0
    else:
        profit_factor = gross_win / abs(gross_loss)

    max_drawdown = _max_drawdown(equity)
    return_pct = float((equity.iloc[-1] / equity.iloc[0]) - 1.0) if not equity.empty and equity.iloc[0] != 0 else 0.0
    cagr = _compute_cagr(equity)
    sharpe = _annualized_sharpe(returns)
    sortino = _annualized_sortino(returns)
    calmar = float(cagr / max_drawdown) if max_drawdown > 0 else (math.inf if cagr > 0 else 0.0)
    exposure_ratio = float(exposure.mean()) if not exposure.empty else 0.0
    date_range = _date_range(equity.index)

    return {
        "trade_count_total": trade_count_total,
        "trades_per_month": trades_per_month,
        "profit_factor": float(profit_factor),
        "expectancy": expectancy,
        "exp_lcb": exp_lcb,
        "max_drawdown": max_drawdown,
        "CAGR": cagr,
        "return_pct": return_pct,
        "exposure_ratio": exposure_ratio,
        "Sharpe_ratio": sharpe,
        "Sortino_ratio": sortino,
        "Calmar_ratio": calmar,
        "duration_days": duration_days,
        "date_range": date_range,
    }


def build_portfolio_equity(returns: pd.Series, initial_capital: float = INITIAL_PORTFOLIO_CAPITAL) -> pd.Series:
    """Convert a return series into an equity curve."""

    returns = returns.astype(float).sort_index()
    if returns.empty:
        return pd.Series(dtype=float)
    growth = (1.0 + returns.fillna(0.0)).cumprod()
    return growth * float(initial_capital)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peaks = equity.cummax()
    drawdown = (equity - peaks) / peaks.replace(0, pd.NA)
    minimum = drawdown.min(skipna=True)
    return abs(float(minimum)) if pd.notna(minimum) else 0.0


def _compute_cagr(equity: pd.Series) -> float:
    if equity.empty or len(equity.index) < 2:
        return 0.0
    duration_days = _duration_days(equity.index)
    if duration_days <= 0 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    years = duration_days / 365.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0


def _annualized_sharpe(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    volatility = float(returns.std(ddof=0))
    if volatility <= 0:
        return 0.0
    mean_return = float(returns.mean())
    return float((mean_return / volatility) * math.sqrt(ANNUAL_BARS_1H))


def _annualized_sortino(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    if downside_std <= 0:
        return 0.0
    mean_return = float(returns.mean())
    return float((mean_return / downside_std) * math.sqrt(ANNUAL_BARS_1H))


def _duration_days(index: pd.Index) -> float:
    if len(index) < 2:
        return 0.0
    timestamps = pd.to_datetime(index, utc=True)
    return float((timestamps[-1] - timestamps[0]).total_seconds() / 86400.0)


def _date_range(index: pd.Index) -> str:
    if len(index) == 0:
        return "n/a"
    timestamps = pd.to_datetime(index, utc=True)
    return f"{timestamps[0].isoformat()}..{timestamps[-1].isoformat()}"
