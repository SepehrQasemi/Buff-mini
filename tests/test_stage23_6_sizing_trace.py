from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.forensics.signal_flow import run_signal_flow_trace
from buffmini.stage23.order_builder import build_adaptive_orders
from buffmini.stage23.rejects import EXECUTION_REJECT_REASONS


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0] * 6,
            "high": [101.0] * 6,
            "low": [99.0] * 6,
            "close": [100.0] * 6,
            "volume": [1000.0] * 6,
            "atr_14": [1.0] * 6,
        }
    )


def _cfg() -> dict:
    return {
        "costs": {"round_trip_cost_pct": 0.1, "slippage_pct": 0.0005},
        "cost_model": {"mode": "simple", "round_trip_cost_pct": 0.1},
        "risk": {"max_gross_exposure": 5.0},
        "evaluation": {
            "stage10": {"evaluation": {"take_profit_atr_multiple": 3.0}},
            "stage23": {
                "enabled": True,
                "order_builder": {
                    "min_stop_atr_mult": 0.8,
                    "min_stop_bps": 8.0,
                    "min_rr": 0.8,
                    "min_trade_notional": 10.0,
                    "allow_size_bump_to_min_notional": True,
                    "rr_fallback_exit_mode": "fixed_atr",
                },
                "execution": {
                    "allow_partial_fill": True,
                    "partial_fill_min_ratio": 0.3,
                    "allow_size_reduction_on_margin_fail": True,
                    "max_size_reduction_steps": 5,
                    "slippage_soft_threshold_bps": 15.0,
                    "slippage_hard_threshold_bps": 40.0,
                },
            },
        },
    }


def test_sizing_trace_has_required_columns() -> None:
    frame = _frame()
    signal = pd.Series([1, 0, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    out = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=_cfg(), symbol="BTC/USDT")
    trace = out["sizing_trace"]
    required = {
        "ts",
        "symbol",
        "side",
        "price",
        "stop_price",
        "raw_size",
        "capped_size",
        "min_notional",
        "rounded_size_before",
        "rounded_size_after",
        "rounding_mode_used",
        "final_notional",
        "decision",
        "reject_reason",
    }
    assert required.issubset(set(trace.columns))
    assert isinstance(out["sizing_trace_summary"], dict)


def test_sizing_trace_reject_reasons_are_known() -> None:
    frame = _frame()
    signal = pd.Series([1, -1, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([0.001, 0.001, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    cfg = _cfg()
    cfg["evaluation"]["stage23"]["order_builder"]["allow_size_bump_to_min_notional"] = False
    out = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg, symbol="ETH/USDT")
    reasons = set(out["sizing_trace"]["reject_reason"].astype(str).tolist())
    reasons.discard("")
    assert reasons.issubset(set(EXECUTION_REJECT_REASONS))


def test_sizing_trace_records_positive_raw_size_for_small_reject() -> None:
    frame = _frame()
    signal = pd.Series([1, 0, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([0.0001, 0.0, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    cfg = _cfg()
    cfg["evaluation"]["stage23"]["order_builder"]["allow_size_bump_to_min_notional"] = False
    out = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg, symbol="BTC/USDT")
    trace = out["sizing_trace"]
    rejected = trace.loc[trace["reject_reason"] == "SIZE_TOO_SMALL"]
    assert not rejected.empty
    assert float(rejected["raw_size"].iloc[0]) > 0.0


def test_trace_runner_writes_sizing_trace_artifacts(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.setdefault("evaluation", {}).setdefault("stage23", {})["enabled"] = True
    result = run_signal_flow_trace(
        config=cfg,
        seed=42,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        mode="classic",
        stages=["classic"],
        families=["price"],
        composers=["none"],
        max_combos=5,
        dry_run=True,
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
    )
    assert (result["trace_dir"] / "sizing_trace.csv").exists()
    assert (result["trace_dir"] / "sizing_trace_summary.json").exists()
