from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.alpha_v2.exits_v2 import trailing_stop_series
from buffmini.backtest.engine import run_backtest
from buffmini.validation.cost_model_v2 import one_way_cost_breakdown_bps
from buffmini.validation.leakage_harness import synthetic_ohlcv


def _frame_with_features(rows: int = 600) -> pd.DataFrame:
    frame = synthetic_ohlcv(rows=rows, seed=123)
    frame["atr_14"] = (
        (pd.to_numeric(frame["high"], errors="coerce") - pd.to_numeric(frame["low"], errors="coerce"))
        .rolling(14, min_periods=1)
        .mean()
        .fillna(1.0)
    )
    frame["signal"] = np.where(np.arange(rows) % 25 == 0, 1, 0)
    return frame


def test_stage17_trailing_stop_monotonic_for_long() -> None:
    frame = _frame_with_features(200)
    stop = trailing_stop_series(
        close=frame["close"],
        atr=frame["atr_14"],
        side=1,
        trailing_k=1.5,
    )
    diffs = np.diff(stop.to_numpy(dtype=float))
    assert (diffs >= -1e-12).all()


def test_stage17_same_candle_priority_deterministic_numpy_vs_pandas() -> None:
    frame = _frame_with_features(400)
    numpy_bt = run_backtest(
        frame=frame,
        strategy_name="t",
        symbol="BTC/USDT",
        signal_col="signal",
        exit_mode="trailing_atr",
        trailing_atr_k=1.5,
        cost_model_cfg={"mode": "simple", "round_trip_cost_pct": 0.1},
        engine_mode="numpy",
    )
    pandas_bt = run_backtest(
        frame=frame,
        strategy_name="t",
        symbol="BTC/USDT",
        signal_col="signal",
        exit_mode="trailing_atr",
        trailing_atr_k=1.5,
        cost_model_cfg={"mode": "simple", "round_trip_cost_pct": 0.1},
        engine_mode="pandas",
    )
    assert int(numpy_bt.metrics.get("trade_count", 0.0)) == int(pandas_bt.metrics.get("trade_count", 0.0))


def test_stage17_drag_stress_increases_per_side_cost_monotonic() -> None:
    frame = _frame_with_features(500)
    base_cfg = {
        "mode": "v2",
        "round_trip_cost_pct": 0.1,
        "v2": {
            "slippage_bps_base": 0.5,
            "slippage_bps_vol_mult": 2.0,
            "spread_bps": 0.5,
            "delay_bars": 0,
            "vol_proxy": "atr_pct",
            "vol_lookback": 14,
            "max_total_bps_per_side": 10.0,
        },
    }
    stress_cfg = {
        "mode": "v2",
        "round_trip_cost_pct": 0.1,
        "v2": {
            "slippage_bps_base": 1.5,
            "slippage_bps_vol_mult": 3.0,
            "spread_bps": 1.5,
            "delay_bars": 1,
            "vol_proxy": "atr_pct",
            "vol_lookback": 14,
            "max_total_bps_per_side": 10.0,
        },
    }
    base_cost = np.mean(
        [one_way_cost_breakdown_bps(frame=frame, bar_index=i, cost_cfg=base_cfg)["total_bps"] for i in range(len(frame))]
    )
    stress_cost = np.mean(
        [one_way_cost_breakdown_bps(frame=frame, bar_index=i, cost_cfg=stress_cfg)["total_bps"] for i in range(len(frame))]
    )
    assert float(stress_cost) >= float(base_cost) - 1e-12
