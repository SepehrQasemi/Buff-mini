from __future__ import annotations

import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, trend_pullback
from buffmini.data.features import calculate_features
from buffmini.utils.hashing import stable_hash
from buffmini.validation.leakage_harness import synthetic_ohlcv


def _frame_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return stable_hash({"rows": 0}, length=16)
    work = frame.copy().reset_index(drop=True)
    if "entry_time" in work.columns:
        work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    if "exit_time" in work.columns:
        work["exit_time"] = pd.to_datetime(work["exit_time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].astype(float).round(12)
    row_hash = pd.util.hash_pandas_object(work, index=False, categorize=False).astype("uint64")
    return stable_hash({"rows": int(len(work)), "sum": int(row_hash.sum())}, length=16)


def test_numpy_fast_path_is_semantically_equivalent_to_pandas() -> None:
    raw = synthetic_ohlcv(rows=24 * 180, seed=123)
    features = calculate_features(raw)
    strategy = trend_pullback()
    features["signal"] = generate_signals(features, strategy=strategy, gating_mode="none")

    common_kwargs = dict(
        frame=features,
        strategy_name="semantic_equivalence",
        symbol="BTC/USDT",
        max_hold_bars=24,
        stop_atr_multiple=1.5,
        take_profit_atr_multiple=3.0,
        round_trip_cost_pct=0.1,
        slippage_pct=0.0005,
        exit_mode="fixed_atr",
    )
    pandas_result = run_backtest(engine_mode="pandas", **common_kwargs)
    numpy_result = run_backtest(engine_mode="numpy", **common_kwargs)
    default_result = run_backtest(**common_kwargs)

    pandas_equity_hash = _frame_hash(pandas_result.equity_curve)
    numpy_equity_hash = _frame_hash(numpy_result.equity_curve)
    pandas_trades_hash = _frame_hash(pandas_result.trades)
    numpy_trades_hash = _frame_hash(numpy_result.trades)
    default_equity_hash = _frame_hash(default_result.equity_curve)
    default_trades_hash = _frame_hash(default_result.trades)

    assert pandas_equity_hash == numpy_equity_hash
    assert pandas_trades_hash == numpy_trades_hash
    assert default_equity_hash == numpy_equity_hash
    assert default_trades_hash == numpy_trades_hash
