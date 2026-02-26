"""Minimal Stage-0 backtest engine."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buffmini.backtest.costs import apply_fee, apply_slippage, round_trip_pct_to_one_way_fee_rate
from buffmini.backtest.metrics import calculate_metrics
from buffmini.types import BacktestResult, Trade


def run_backtest(
    frame: pd.DataFrame,
    strategy_name: str,
    symbol: str,
    signal_col: str = "signal",
    atr_col: str = "atr_14",
    initial_capital: float = 10_000.0,
    stop_atr_multiple: float = 1.5,
    take_profit_atr_multiple: float = 3.0,
    max_hold_bars: int = 24,
    round_trip_cost_pct: float = 0.1,
    slippage_pct: float = 0.0005,
) -> BacktestResult:
    """Run a single-position long/short backtest with ATR exits."""

    required = {"timestamp", "high", "low", "close", signal_col, atr_col}
    missing = required.difference(frame.columns)
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    data = frame.copy().reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

    one_way_fee_rate = round_trip_pct_to_one_way_fee_rate(round_trip_cost_pct)

    equity = float(initial_capital)
    equity_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    position: dict[str, Any] | None = None

    for idx, row in data.iterrows():
        timestamp = row["timestamp"]
        signal = int(row[signal_col]) if pd.notna(row[signal_col]) else 0
        atr = float(row[atr_col]) if pd.notna(row[atr_col]) else float("nan")

        if position is None and signal in (-1, 1) and pd.notna(atr) and atr > 0 and equity > 0:
            side = "long" if signal == 1 else "short"
            entry_exec_side = "buy" if side == "long" else "sell"
            entry_price = apply_slippage(float(row["close"]), slippage_pct, entry_exec_side)
            qty = equity / entry_price if entry_price > 0 else 0.0
            if qty <= 0:
                equity_rows.append({"timestamp": timestamp, "equity": equity})
                continue

            entry_notional = entry_price * qty
            entry_fee = apply_fee(entry_notional, one_way_fee_rate)
            equity -= entry_fee

            stop_price = (
                entry_price - stop_atr_multiple * atr
                if side == "long"
                else entry_price + stop_atr_multiple * atr
            )
            tp_price = (
                entry_price + take_profit_atr_multiple * atr
                if side == "long"
                else entry_price - take_profit_atr_multiple * atr
            )

            position = {
                "entry_idx": idx,
                "entry_time": timestamp,
                "entry_price": entry_price,
                "side": side,
                "qty": qty,
                "entry_notional": entry_notional,
                "entry_fee": entry_fee,
                "stop_price": stop_price,
                "tp_price": tp_price,
                "just_opened": True,
            }

        if position is not None:
            if position["just_opened"]:
                position["just_opened"] = False
                equity_rows.append({"timestamp": timestamp, "equity": equity})
                continue

            side = position["side"]
            exit_reason = ""
            exit_price = None

            if side == "long":
                if float(row["low"]) <= position["stop_price"]:
                    exit_reason = "stop_loss"
                    exit_price = apply_slippage(position["stop_price"], slippage_pct, "sell")
                elif float(row["high"]) >= position["tp_price"]:
                    exit_reason = "take_profit"
                    exit_price = apply_slippage(position["tp_price"], slippage_pct, "sell")
            else:
                if float(row["high"]) >= position["stop_price"]:
                    exit_reason = "stop_loss"
                    exit_price = apply_slippage(position["stop_price"], slippage_pct, "buy")
                elif float(row["low"]) <= position["tp_price"]:
                    exit_reason = "take_profit"
                    exit_price = apply_slippage(position["tp_price"], slippage_pct, "buy")

            bars_held = idx - int(position["entry_idx"])
            if not exit_reason and bars_held >= max_hold_bars:
                exit_reason = "time_stop"
                exec_side = "sell" if side == "long" else "buy"
                exit_price = apply_slippage(float(row["close"]), slippage_pct, exec_side)

            if exit_reason and exit_price is not None:
                direction = 1.0 if side == "long" else -1.0
                qty = float(position["qty"])
                entry_price = float(position["entry_price"])
                entry_notional = float(position["entry_notional"])
                entry_fee = float(position["entry_fee"])
                exit_notional = float(exit_price) * qty
                exit_fee = apply_fee(exit_notional, one_way_fee_rate)

                gross_pnl = (float(exit_price) - entry_price) * qty * direction
                net_trade_pnl = gross_pnl - entry_fee - exit_fee
                equity += gross_pnl - exit_fee

                trade = Trade(
                    strategy=strategy_name,
                    symbol=symbol,
                    side=side,
                    entry_time=position["entry_time"],
                    exit_time=timestamp,
                    entry_price=entry_price,
                    exit_price=float(exit_price),
                    pnl=float(net_trade_pnl),
                    return_pct=float(net_trade_pnl / entry_notional) if entry_notional else 0.0,
                    bars_held=bars_held,
                    exit_reason=exit_reason,
                )
                trades.append(trade.to_dict())
                position = None

        equity_rows.append({"timestamp": timestamp, "equity": equity})

    if position is not None:
        final_row = data.iloc[-1]
        final_time = pd.to_datetime(final_row["timestamp"], utc=True)
        side = position["side"]
        exec_side = "sell" if side == "long" else "buy"
        exit_price = apply_slippage(float(final_row["close"]), slippage_pct, exec_side)

        direction = 1.0 if side == "long" else -1.0
        qty = float(position["qty"])
        entry_price = float(position["entry_price"])
        entry_notional = float(position["entry_notional"])
        entry_fee = float(position["entry_fee"])
        exit_notional = exit_price * qty
        exit_fee = apply_fee(exit_notional, one_way_fee_rate)

        gross_pnl = (exit_price - entry_price) * qty * direction
        net_trade_pnl = gross_pnl - entry_fee - exit_fee
        equity += gross_pnl - exit_fee

        trade = Trade(
            strategy=strategy_name,
            symbol=symbol,
            side=side,
            entry_time=position["entry_time"],
            exit_time=final_time,
            entry_price=entry_price,
            exit_price=float(exit_price),
            pnl=float(net_trade_pnl),
            return_pct=float(net_trade_pnl / entry_notional) if entry_notional else 0.0,
            bars_held=int(len(data) - 1 - position["entry_idx"]),
            exit_reason="end_of_data",
        )
        trades.append(trade.to_dict())

        if equity_rows:
            equity_rows[-1]["equity"] = equity

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.DataFrame(equity_rows)
    metrics = calculate_metrics(trades_df, equity_curve)

    return BacktestResult(trades=trades_df, equity_curve=equity_curve, metrics=metrics)
