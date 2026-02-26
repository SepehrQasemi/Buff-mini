"""Stage-0.6 strategy, gating, window, and report sanity tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, stage06_strategies
from buffmini.config import load_config
from buffmini.data.features import calculate_features
from buffmini.data.window import slice_last_n_months


def _make_filter_test_frame(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")

    base = 100 + np.sin(np.linspace(0, 40, rows)) * 3
    vol_state = np.where(np.arange(rows) % 80 < 40, 0.2, 2.0)
    close = base + rng.normal(0, vol_state, rows)
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    candle_spread = np.abs(rng.normal(0.6, 0.2, rows)) * vol_state
    high = np.maximum(open_, close) + candle_spread
    low = np.minimum(open_, close) - candle_spread

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1_000, 2_000, rows),
        }
    )


def test_stage06_returns_exactly_five_strategies_with_required_fields() -> None:
    strategies = stage06_strategies()
    assert len(strategies) == 5

    names = [strategy.name for strategy in strategies]
    expected = {
        "Donchian Breakout",
        "RSI Mean Reversion",
        "Trend Pullback",
        "Bollinger Mean Reversion",
        "Range Breakout w/ EMA Trend Filter",
    }
    assert set(names) == expected

    for strategy in strategies:
        assert strategy.entry_rules
        assert strategy.exit_rules
        assert isinstance(strategy.parameters, dict)


def test_window_slicing_last_n_months_selects_expected_range() -> None:
    timestamps = pd.date_range("2020-01-01", periods=72, freq="MS", tz="UTC")
    frame = pd.DataFrame({"timestamp": timestamps, "value": np.arange(len(timestamps))})

    sliced, date_range = slice_last_n_months(frame, window_months=12, end_mode="latest")

    expected_end = timestamps.max()
    expected_start = expected_end - pd.DateOffset(months=12)

    sliced_ts = pd.to_datetime(sliced["timestamp"], utc=True)
    assert sliced_ts.iloc[-1] == expected_end
    assert sliced_ts.iloc[0] >= expected_start
    assert len(sliced) == 13
    assert ".." in date_range


def test_stage06_gating_reduces_trade_count_for_at_least_one_strategy() -> None:
    frame = _make_filter_test_frame()
    features = calculate_features(frame)

    reduced = False
    for strategy in stage06_strategies():
        ungated = features.copy()
        ungated["signal"] = generate_signals(ungated, strategy, gating_mode="none")
        ungated_result = run_backtest(
            frame=ungated,
            strategy_name=strategy.name,
            symbol="BTC/USDT",
            round_trip_cost_pct=0.1,
            slippage_pct=0.0,
        )

        gated = features.copy()
        gated["signal"] = generate_signals(gated, strategy, gating_mode="vol+regime")
        gated_result = run_backtest(
            frame=gated,
            strategy_name=strategy.name,
            symbol="BTC/USDT",
            round_trip_cost_pct=0.1,
            slippage_pct=0.0,
        )

        if gated_result.metrics["trade_count"] < ungated_result.metrics["trade_count"]:
            reduced = True
            break

    assert reduced


def test_stage06_report_json_sanity_from_offline_matrix(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    base_config = load_config(root / "configs" / "default.yaml")

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    costs = [1.0, 0.2, 0.1]
    payload = {"runs": []}

    for cost in costs:
        cfg = json.loads(json.dumps(base_config))
        cfg["costs"]["round_trip_cost_pct"] = cost
        cfg_path = tmp_path / f"cfg_{cost}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        run_id = f"cost_{str(cost).replace('.', '_')}"
        cmd = [
            sys.executable,
            "scripts/run_stage0.py",
            "--stage06",
            "--dry-run",
            "--gating",
            "vol+regime",
            "--synthetic-bars",
            "260",
            "--config",
            str(cfg_path),
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            run_id,
        ]
        subprocess.run(cmd, cwd=root, check=True)

        leaderboard = pd.read_csv(runs_dir / run_id / "leaderboard.csv")
        results = [
            {
                "symbol": str(row["symbol"]),
                "strategy": str(row["strategy"]),
                "profit_factor": float(row["profit_factor"]),
                "expectancy": float(row["expectancy"]),
                "max_drawdown": float(row["max_drawdown"]),
                "trade_count": float(row["trade_count"]),
                "final_equity": float(row["final_equity"]),
            }
            for _, row in leaderboard.iterrows()
        ]
        payload["runs"].append(
            {
                "round_trip_cost_pct": cost,
                "run_id": run_id,
                "results": results,
            }
        )

    observed_costs = {float(run["round_trip_cost_pct"]) for run in payload["runs"]}
    assert observed_costs == {1.0, 0.2, 0.1}

    observed_symbols = {
        result["symbol"]
        for run in payload["runs"]
        for result in run["results"]
    }
    assert {"BTC/USDT", "ETH/USDT"}.issubset(observed_symbols)

    assert all(run["results"] for run in payload["runs"])
