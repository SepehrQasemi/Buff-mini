"""Minimal backtest engine with configurable exit modes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.costs import (
    apply_fee,
    apply_slippage,
    fill_price_and_index,
    normalized_cost_cfg,
    one_way_slippage_for_bar,
    round_trip_pct_to_one_way_fee_rate,
)
from buffmini.backtest.metrics import calculate_metrics
from buffmini.types import BacktestResult, Trade


_ALLOWED_EXIT_MODES = {"fixed_atr", "breakeven_1r", "trailing_atr", "partial_then_trail"}
_ALLOWED_ENGINE_MODES = {"numpy", "pandas"}


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
    exit_mode: str = "fixed_atr",
    trailing_atr_k: float = 1.5,
    partial_size: float = 0.5,
    cost_model_cfg: dict[str, Any] | None = None,
    engine_mode: str = "numpy",
) -> BacktestResult:
    """Run a single-position long/short backtest with multiple exit styles."""

    if str(exit_mode) not in _ALLOWED_EXIT_MODES:
        raise ValueError(f"Unsupported exit_mode: {exit_mode}")
    resolved_engine_mode = str(engine_mode).strip().lower()
    if resolved_engine_mode not in _ALLOWED_ENGINE_MODES:
        raise ValueError(f"Unsupported engine_mode: {engine_mode}")
    if resolved_engine_mode == "pandas":
        return _run_backtest_pandas(
            frame=frame,
            strategy_name=strategy_name,
            symbol=symbol,
            signal_col=signal_col,
            atr_col=atr_col,
            initial_capital=initial_capital,
            stop_atr_multiple=stop_atr_multiple,
            take_profit_atr_multiple=take_profit_atr_multiple,
            max_hold_bars=max_hold_bars,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            exit_mode=exit_mode,
            trailing_atr_k=trailing_atr_k,
            partial_size=partial_size,
            cost_model_cfg=cost_model_cfg,
        )
    return _run_backtest_numpy(
        frame=frame,
        strategy_name=strategy_name,
        symbol=symbol,
        signal_col=signal_col,
        atr_col=atr_col,
        initial_capital=initial_capital,
        stop_atr_multiple=stop_atr_multiple,
        take_profit_atr_multiple=take_profit_atr_multiple,
        max_hold_bars=max_hold_bars,
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
        exit_mode=exit_mode,
        trailing_atr_k=trailing_atr_k,
        partial_size=partial_size,
        cost_model_cfg=cost_model_cfg,
    )


def _validate_backtest_frame(frame: pd.DataFrame, signal_col: str, atr_col: str) -> pd.DataFrame:
    required = {"timestamp", "high", "low", "close", signal_col, atr_col}
    missing = required.difference(frame.columns)
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        raise ValueError(msg)
    data = frame.copy().reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data


def _run_backtest_pandas(
    *,
    frame: pd.DataFrame,
    strategy_name: str,
    symbol: str,
    signal_col: str,
    atr_col: str,
    initial_capital: float,
    stop_atr_multiple: float,
    take_profit_atr_multiple: float,
    max_hold_bars: int,
    round_trip_cost_pct: float,
    slippage_pct: float,
    exit_mode: str,
    trailing_atr_k: float,
    partial_size: float,
    cost_model_cfg: dict[str, Any] | None,
) -> BacktestResult:
    data = _validate_backtest_frame(frame, signal_col=signal_col, atr_col=atr_col)

    cost_cfg = normalized_cost_cfg(
        round_trip_cost_pct=float(round_trip_cost_pct),
        slippage_pct=float(slippage_pct),
        cost_model_cfg=cost_model_cfg,
    )
    one_way_fee_rate = round_trip_pct_to_one_way_fee_rate(float(cost_cfg["round_trip_cost_pct"]))

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
            entry_price, entry_idx = _execution_fill_price(
                data=data,
                trigger_index=idx,
                base_price=float(row["close"]),
                side=entry_exec_side,
                cost_cfg=cost_cfg,
                atr_col=atr_col,
            )
            entry_time = pd.to_datetime(data.iloc[entry_idx]["timestamp"], utc=True)
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
                "entry_idx": int(entry_idx),
                "entry_time": entry_time,
                "entry_price": entry_price,
                "side": side,
                "qty_total": qty,
                "qty_open": qty,
                "entry_notional": entry_notional,
                "entry_fee": entry_fee,
                "atr_entry": atr,
                "stop_price": stop_price,
                "tp_price": tp_price,
                "just_opened": True,
                "realized_partial_pnl": 0.0,
                "partial_taken": False,
                "breakeven_applied": False,
                "highest_price": float(row["high"]),
                "lowest_price": float(row["low"]),
            }

        if position is not None:
            if position["just_opened"]:
                position["just_opened"] = False
                equity_rows.append({"timestamp": timestamp, "equity": equity})
                continue

            side = position["side"]
            direction = 1.0 if side == "long" else -1.0
            atr_current = float(row[atr_col]) if pd.notna(row[atr_col]) and float(row[atr_col]) > 0 else float(position["atr_entry"])
            one_r = float(position["atr_entry"]) * float(stop_atr_multiple)
            one_r_price = (
                float(position["entry_price"]) + one_r
                if side == "long"
                else float(position["entry_price"]) - one_r
            )

            if exit_mode in {"breakeven_1r", "partial_then_trail"} and not position["breakeven_applied"]:
                reached_one_r = (
                    float(row["high"]) >= one_r_price
                    if side == "long"
                    else float(row["low"]) <= one_r_price
                )
                if reached_one_r:
                    if side == "long":
                        position["stop_price"] = max(float(position["stop_price"]), float(position["entry_price"]))
                    else:
                        position["stop_price"] = min(float(position["stop_price"]), float(position["entry_price"]))
                    position["breakeven_applied"] = True

            if exit_mode == "partial_then_trail" and not position["partial_taken"]:
                reached_one_r = (
                    float(row["high"]) >= one_r_price
                    if side == "long"
                    else float(row["low"]) <= one_r_price
                )
                if reached_one_r:
                    partial_qty = float(position["qty_total"]) * float(partial_size)
                    partial_qty = max(0.0, min(partial_qty, float(position["qty_open"])))
                    if partial_qty > 0:
                        partial_side = "sell" if side == "long" else "buy"
                        partial_price, _ = _execution_fill_price(
                            data=data,
                            trigger_index=idx,
                            base_price=float(one_r_price),
                            side=partial_side,
                            cost_cfg=cost_cfg,
                            atr_col=atr_col,
                        )
                        partial_notional = partial_price * partial_qty
                        partial_fee = apply_fee(partial_notional, one_way_fee_rate)
                        partial_gross = (partial_price - float(position["entry_price"])) * partial_qty * direction
                        partial_net = partial_gross - partial_fee

                        position["qty_open"] = float(position["qty_open"]) - partial_qty
                        position["realized_partial_pnl"] = float(position["realized_partial_pnl"]) + partial_net
                        position["partial_taken"] = True
                        equity += partial_net

            if exit_mode in {"trailing_atr", "partial_then_trail"}:
                if side == "long":
                    position["highest_price"] = max(float(position["highest_price"]), float(row["high"]))
                    trail_stop = float(position["highest_price"]) - float(trailing_atr_k) * atr_current
                    position["stop_price"] = max(float(position["stop_price"]), trail_stop)
                else:
                    position["lowest_price"] = min(float(position["lowest_price"]), float(row["low"]))
                    trail_stop = float(position["lowest_price"]) + float(trailing_atr_k) * atr_current
                    position["stop_price"] = min(float(position["stop_price"]), trail_stop)

            exit_reason = ""
            exit_price = None
            exit_idx = int(idx)
            exit_time = timestamp

            if side == "long":
                stop_hit = float(row["low"]) <= float(position["stop_price"])
                tp_hit = float(row["high"]) >= float(position["tp_price"])
            else:
                stop_hit = float(row["high"]) >= float(position["stop_price"])
                tp_hit = float(row["low"]) <= float(position["tp_price"])

            if stop_hit:
                exit_reason = "stop_loss"
                exec_side = "sell" if side == "long" else "buy"
                exit_price, exit_idx = _execution_fill_price(
                    data=data,
                    trigger_index=idx,
                    base_price=float(position["stop_price"]),
                    side=exec_side,
                    cost_cfg=cost_cfg,
                    atr_col=atr_col,
                )
                exit_time = pd.to_datetime(data.iloc[exit_idx]["timestamp"], utc=True)
            elif exit_mode in {"fixed_atr", "breakeven_1r"} and tp_hit:
                exit_reason = "take_profit"
                exec_side = "sell" if side == "long" else "buy"
                exit_price, exit_idx = _execution_fill_price(
                    data=data,
                    trigger_index=idx,
                    base_price=float(position["tp_price"]),
                    side=exec_side,
                    cost_cfg=cost_cfg,
                    atr_col=atr_col,
                )
                exit_time = pd.to_datetime(data.iloc[exit_idx]["timestamp"], utc=True)

            bars_held = idx - int(position["entry_idx"])
            if not exit_reason and bars_held >= int(max_hold_bars):
                exit_reason = "time_stop"
                exec_side = "sell" if side == "long" else "buy"
                exit_price, exit_idx = _execution_fill_price(
                    data=data,
                    trigger_index=idx,
                    base_price=float(row["close"]),
                    side=exec_side,
                    cost_cfg=cost_cfg,
                    atr_col=atr_col,
                )
                exit_time = pd.to_datetime(data.iloc[exit_idx]["timestamp"], utc=True)

            if exit_reason and exit_price is not None:
                qty_open = float(position["qty_open"])
                if qty_open <= 0:
                    qty_open = 0.0

                exit_notional = float(exit_price) * qty_open
                exit_fee = apply_fee(exit_notional, one_way_fee_rate)
                gross_pnl = (float(exit_price) - float(position["entry_price"])) * qty_open * direction
                exit_net = gross_pnl - exit_fee

                equity += exit_net
                net_trade_pnl = (
                    float(position["realized_partial_pnl"])
                    + exit_net
                    - float(position["entry_fee"])
                )

                trade = Trade(
                    strategy=strategy_name,
                    symbol=symbol,
                    side=side,
                    entry_time=position["entry_time"],
                    exit_time=exit_time,
                    entry_price=float(position["entry_price"]),
                    exit_price=float(exit_price),
                    pnl=float(net_trade_pnl),
                    return_pct=float(net_trade_pnl / float(position["entry_notional"])) if position["entry_notional"] else 0.0,
                    bars_held=int(max(0, exit_idx - int(position["entry_idx"]))),
                    exit_reason=exit_reason,
                )
                trades.append(trade.to_dict())
                position = None

        equity_rows.append({"timestamp": timestamp, "equity": equity})

    if position is not None:
        final_idx = len(data) - 1
        final_row = data.iloc[final_idx]
        side = position["side"]
        direction = 1.0 if side == "long" else -1.0
        exec_side = "sell" if side == "long" else "buy"
        exit_price, exit_idx = _execution_fill_price(
            data=data,
            trigger_index=final_idx,
            base_price=float(final_row["close"]),
            side=exec_side,
            cost_cfg=cost_cfg,
            atr_col=atr_col,
        )
        final_time = pd.to_datetime(data.iloc[exit_idx]["timestamp"], utc=True)

        qty_open = float(position["qty_open"])
        exit_notional = exit_price * qty_open
        exit_fee = apply_fee(exit_notional, one_way_fee_rate)
        gross_pnl = (exit_price - float(position["entry_price"])) * qty_open * direction
        exit_net = gross_pnl - exit_fee
        equity += exit_net

        net_trade_pnl = (
            float(position["realized_partial_pnl"])
            + exit_net
            - float(position["entry_fee"])
        )

        trade = Trade(
            strategy=strategy_name,
            symbol=symbol,
            side=side,
            entry_time=position["entry_time"],
            exit_time=final_time,
            entry_price=float(position["entry_price"]),
            exit_price=float(exit_price),
            pnl=float(net_trade_pnl),
            return_pct=float(net_trade_pnl / float(position["entry_notional"])) if position["entry_notional"] else 0.0,
            bars_held=int(max(0, exit_idx - int(position["entry_idx"]))),
            exit_reason="end_of_data",
        )
        trades.append(trade.to_dict())

        if equity_rows:
            equity_rows[-1]["equity"] = equity

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.DataFrame(equity_rows)
    metrics = calculate_metrics(trades_df, equity_curve)
    return BacktestResult(trades=trades_df, equity_curve=equity_curve, metrics=metrics)


def _run_backtest_numpy(
    *,
    frame: pd.DataFrame,
    strategy_name: str,
    symbol: str,
    signal_col: str,
    atr_col: str,
    initial_capital: float,
    stop_atr_multiple: float,
    take_profit_atr_multiple: float,
    max_hold_bars: int,
    round_trip_cost_pct: float,
    slippage_pct: float,
    exit_mode: str,
    trailing_atr_k: float,
    partial_size: float,
    cost_model_cfg: dict[str, Any] | None,
) -> BacktestResult:
    data = _validate_backtest_frame(frame, signal_col=signal_col, atr_col=atr_col)

    ts = pd.to_datetime(data["timestamp"], utc=True)
    high = pd.to_numeric(data["high"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(data["low"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    close = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    atr = pd.to_numeric(data[atr_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    signal = (
        pd.to_numeric(data[signal_col], errors="coerce")
        .fillna(0)
        .astype(np.int64)
        .to_numpy(copy=False)
    )

    cost_cfg = normalized_cost_cfg(
        round_trip_cost_pct=float(round_trip_cost_pct),
        slippage_pct=float(slippage_pct),
        cost_model_cfg=cost_model_cfg,
    )
    one_way_fee_rate = round_trip_pct_to_one_way_fee_rate(float(cost_cfg["round_trip_cost_pct"]))

    equity = float(initial_capital)
    equity_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None

    for idx in range(len(data)):
        timestamp = ts.iloc[idx]
        sig = int(signal[idx]) if np.isfinite(signal[idx]) else 0
        atr_val = float(atr[idx]) if np.isfinite(atr[idx]) else float("nan")

        if position is None and sig in (-1, 1) and np.isfinite(atr_val) and atr_val > 0 and equity > 0:
            side = "long" if sig == 1 else "short"
            entry_exec_side = "buy" if side == "long" else "sell"
            entry_price, entry_idx = _execution_fill_price(
                data=data,
                trigger_index=idx,
                base_price=float(close[idx]),
                side=entry_exec_side,
                cost_cfg=cost_cfg,
                atr_col=atr_col,
            )
            entry_time = pd.to_datetime(ts.iloc[entry_idx], utc=True)
            qty = equity / entry_price if entry_price > 0 else 0.0
            if qty <= 0:
                equity_rows.append({"timestamp": timestamp, "equity": equity})
                continue

            entry_notional = entry_price * qty
            entry_fee = apply_fee(entry_notional, one_way_fee_rate)
            equity -= entry_fee

            stop_price = (
                entry_price - stop_atr_multiple * atr_val
                if side == "long"
                else entry_price + stop_atr_multiple * atr_val
            )
            tp_price = (
                entry_price + take_profit_atr_multiple * atr_val
                if side == "long"
                else entry_price - take_profit_atr_multiple * atr_val
            )

            position = {
                "entry_idx": int(entry_idx),
                "entry_time": entry_time,
                "entry_price": float(entry_price),
                "side": side,
                "qty_total": float(qty),
                "qty_open": float(qty),
                "entry_notional": float(entry_notional),
                "entry_fee": float(entry_fee),
                "atr_entry": float(atr_val),
                "stop_price": float(stop_price),
                "tp_price": float(tp_price),
                "just_opened": True,
                "realized_partial_pnl": 0.0,
                "partial_taken": False,
                "breakeven_applied": False,
                "highest_price": float(high[idx]),
                "lowest_price": float(low[idx]),
            }

        if position is not None:
            if position["just_opened"]:
                position["just_opened"] = False
                equity_rows.append({"timestamp": timestamp, "equity": equity})
                continue

            side = str(position["side"])
            direction = 1.0 if side == "long" else -1.0
            atr_current = float(atr[idx]) if np.isfinite(atr[idx]) and float(atr[idx]) > 0 else float(position["atr_entry"])
            one_r = float(position["atr_entry"]) * float(stop_atr_multiple)
            one_r_price = float(position["entry_price"]) + one_r if side == "long" else float(position["entry_price"]) - one_r

            if exit_mode in {"breakeven_1r", "partial_then_trail"} and not bool(position["breakeven_applied"]):
                reached_one_r = float(high[idx]) >= one_r_price if side == "long" else float(low[idx]) <= one_r_price
                if reached_one_r:
                    if side == "long":
                        position["stop_price"] = max(float(position["stop_price"]), float(position["entry_price"]))
                    else:
                        position["stop_price"] = min(float(position["stop_price"]), float(position["entry_price"]))
                    position["breakeven_applied"] = True

            if exit_mode == "partial_then_trail" and not bool(position["partial_taken"]):
                reached_one_r = float(high[idx]) >= one_r_price if side == "long" else float(low[idx]) <= one_r_price
                if reached_one_r:
                    partial_qty = float(position["qty_total"]) * float(partial_size)
                    partial_qty = max(0.0, min(partial_qty, float(position["qty_open"])))
                    if partial_qty > 0:
                        partial_side = "sell" if side == "long" else "buy"
                        partial_price, _ = _execution_fill_price(
                            data=data,
                            trigger_index=idx,
                            base_price=float(one_r_price),
                            side=partial_side,
                            cost_cfg=cost_cfg,
                            atr_col=atr_col,
                        )
                        partial_notional = float(partial_price) * partial_qty
                        partial_fee = apply_fee(partial_notional, one_way_fee_rate)
                        partial_gross = (float(partial_price) - float(position["entry_price"])) * partial_qty * direction
                        partial_net = partial_gross - partial_fee
                        position["qty_open"] = float(position["qty_open"]) - partial_qty
                        position["realized_partial_pnl"] = float(position["realized_partial_pnl"]) + partial_net
                        position["partial_taken"] = True
                        equity += partial_net

            if exit_mode in {"trailing_atr", "partial_then_trail"}:
                if side == "long":
                    position["highest_price"] = max(float(position["highest_price"]), float(high[idx]))
                    trail_stop = float(position["highest_price"]) - float(trailing_atr_k) * atr_current
                    position["stop_price"] = max(float(position["stop_price"]), trail_stop)
                else:
                    position["lowest_price"] = min(float(position["lowest_price"]), float(low[idx]))
                    trail_stop = float(position["lowest_price"]) + float(trailing_atr_k) * atr_current
                    position["stop_price"] = min(float(position["stop_price"]), trail_stop)

            exit_reason = ""
            exit_price = None
            exit_idx = int(idx)
            exit_time = timestamp

            if side == "long":
                stop_hit = float(low[idx]) <= float(position["stop_price"])
                tp_hit = float(high[idx]) >= float(position["tp_price"])
            else:
                stop_hit = float(high[idx]) >= float(position["stop_price"])
                tp_hit = float(low[idx]) <= float(position["tp_price"])

            if stop_hit:
                exit_reason = "stop_loss"
                exec_side = "sell" if side == "long" else "buy"
                exit_price, exit_idx = _execution_fill_price(
                    data=data,
                    trigger_index=idx,
                    base_price=float(position["stop_price"]),
                    side=exec_side,
                    cost_cfg=cost_cfg,
                    atr_col=atr_col,
                )
                exit_time = pd.to_datetime(ts.iloc[exit_idx], utc=True)
            elif exit_mode in {"fixed_atr", "breakeven_1r"} and tp_hit:
                exit_reason = "take_profit"
                exec_side = "sell" if side == "long" else "buy"
                exit_price, exit_idx = _execution_fill_price(
                    data=data,
                    trigger_index=idx,
                    base_price=float(position["tp_price"]),
                    side=exec_side,
                    cost_cfg=cost_cfg,
                    atr_col=atr_col,
                )
                exit_time = pd.to_datetime(ts.iloc[exit_idx], utc=True)

            bars_held = int(idx) - int(position["entry_idx"])
            if not exit_reason and bars_held >= int(max_hold_bars):
                exit_reason = "time_stop"
                exec_side = "sell" if side == "long" else "buy"
                exit_price, exit_idx = _execution_fill_price(
                    data=data,
                    trigger_index=idx,
                    base_price=float(close[idx]),
                    side=exec_side,
                    cost_cfg=cost_cfg,
                    atr_col=atr_col,
                )
                exit_time = pd.to_datetime(ts.iloc[exit_idx], utc=True)

            if exit_reason and exit_price is not None:
                qty_open = float(position["qty_open"])
                if qty_open <= 0:
                    qty_open = 0.0
                exit_notional = float(exit_price) * qty_open
                exit_fee = apply_fee(exit_notional, one_way_fee_rate)
                gross_pnl = (float(exit_price) - float(position["entry_price"])) * qty_open * direction
                exit_net = gross_pnl - exit_fee
                equity += exit_net

                net_trade_pnl = float(position["realized_partial_pnl"]) + exit_net - float(position["entry_fee"])
                trade = Trade(
                    strategy=strategy_name,
                    symbol=symbol,
                    side=side,
                    entry_time=position["entry_time"],
                    exit_time=exit_time,
                    entry_price=float(position["entry_price"]),
                    exit_price=float(exit_price),
                    pnl=float(net_trade_pnl),
                    return_pct=float(net_trade_pnl / float(position["entry_notional"])) if position["entry_notional"] else 0.0,
                    bars_held=int(max(0, int(exit_idx) - int(position["entry_idx"]))),
                    exit_reason=exit_reason,
                )
                trades.append(trade.to_dict())
                position = None

        equity_rows.append({"timestamp": timestamp, "equity": equity})

    if position is not None:
        final_idx = len(data) - 1
        side = str(position["side"])
        direction = 1.0 if side == "long" else -1.0
        exec_side = "sell" if side == "long" else "buy"
        exit_price, exit_idx = _execution_fill_price(
            data=data,
            trigger_index=final_idx,
            base_price=float(close[final_idx]),
            side=exec_side,
            cost_cfg=cost_cfg,
            atr_col=atr_col,
        )
        final_time = pd.to_datetime(ts.iloc[exit_idx], utc=True)
        qty_open = float(position["qty_open"])
        exit_notional = float(exit_price) * qty_open
        exit_fee = apply_fee(exit_notional, one_way_fee_rate)
        gross_pnl = (float(exit_price) - float(position["entry_price"])) * qty_open * direction
        exit_net = gross_pnl - exit_fee
        equity += exit_net

        net_trade_pnl = float(position["realized_partial_pnl"]) + exit_net - float(position["entry_fee"])
        trade = Trade(
            strategy=strategy_name,
            symbol=symbol,
            side=side,
            entry_time=position["entry_time"],
            exit_time=final_time,
            entry_price=float(position["entry_price"]),
            exit_price=float(exit_price),
            pnl=float(net_trade_pnl),
            return_pct=float(net_trade_pnl / float(position["entry_notional"])) if position["entry_notional"] else 0.0,
            bars_held=int(max(0, int(exit_idx) - int(position["entry_idx"]))),
            exit_reason="end_of_data",
        )
        trades.append(trade.to_dict())
        if equity_rows:
            equity_rows[-1]["equity"] = equity

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.DataFrame(equity_rows)
    metrics = calculate_metrics(trades_df, equity_curve)
    return BacktestResult(trades=trades_df, equity_curve=equity_curve, metrics=metrics)


def _execution_fill_price(
    data: pd.DataFrame,
    trigger_index: int,
    base_price: float,
    side: str,
    cost_cfg: dict[str, Any],
    atr_col: str,
) -> tuple[float, int]:
    """Resolve delayed fill and apply one-way slippage/spread."""

    fill_base_price, fill_idx = fill_price_and_index(
        frame=data,
        trigger_index=int(trigger_index),
        base_price=float(base_price),
        cost_cfg=cost_cfg,
    )
    slippage_rate = one_way_slippage_for_bar(
        frame=data,
        bar_index=int(fill_idx),
        cost_cfg=cost_cfg,
        atr_col=atr_col,
        close_col="close",
    )
    filled = apply_slippage(float(fill_base_price), float(slippage_rate), "buy" if side == "buy" else "sell")
    return float(filled), int(fill_idx)
