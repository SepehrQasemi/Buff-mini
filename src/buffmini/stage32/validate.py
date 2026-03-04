"""Nested walk-forward and MC-lite validation for Stage-32 finalists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.stage26.conditional_eval import bootstrap_lcb
from buffmini.stage31.dsl import DSLStrategy, evaluate_strategy
from buffmini.validation.walkforward_v2 import build_windows


@dataclass(frozen=True)
class ValidationConfig:
    train_days: int = 180
    holdout_days: int = 30
    forward_days: int = 30
    step_days: int = 30
    min_trades_window: int = 8
    min_exposure_window: float = 0.01
    min_usable_windows: int = 3
    mc_min_trades: int = 30
    seed: int = 42


def validate_candidates(
    *,
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    symbol: str,
    timeframe: str,
    cfg: ValidationConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    conf = cfg or ValidationConfig()
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
    if ts.empty or candidates.empty:
        empty = pd.DataFrame()
        return empty, {"wf_executed_pct": 0.0, "mc_trigger_pct": 0.0, "validated_count": 0}
    windows = build_windows(
        start_ts=ts.iloc[0],
        end_ts=ts.iloc[-1],
        train_days=int(conf.train_days),
        holdout_days=int(conf.holdout_days),
        forward_days=int(conf.forward_days),
        step_days=int(conf.step_days),
        reserve_tail_days=0,
    )
    rows: list[dict[str, Any]] = []
    for item in candidates.to_dict(orient="records"):
        strategy = item.get("strategy")
        if not isinstance(strategy, DSLStrategy):
            continue
        pooled: list[float] = []
        usable_windows = 0
        total_windows = 0
        for window in windows:
            forward_mask = (pd.to_datetime(frame["timestamp"], utc=True, errors="coerce") >= window.forward_start) & (
                pd.to_datetime(frame["timestamp"], utc=True, errors="coerce") < window.forward_end
            )
            forward = frame.loc[forward_mask].reset_index(drop=True)
            if forward.shape[0] < 50:
                continue
            total_windows += 1
            signal = evaluate_strategy(strategy, forward)
            result = run_backtest(
                frame=forward.assign(signal=signal),
                strategy_name=f"Stage32::{strategy.name}",
                symbol=str(symbol),
                signal_col="signal",
                stop_atr_multiple=float(getattr(strategy, "stop_atr_multiple", 1.5)),
                take_profit_atr_multiple=float(getattr(strategy, "take_profit_atr_multiple", 3.0)),
                max_hold_bars=int(getattr(strategy, "max_hold_bars", 24)),
                round_trip_cost_pct=0.1,
                slippage_pct=0.0005,
                exit_mode=str(getattr(strategy, "exit_mode", "fixed_atr")),
                cost_model_cfg={},
            )
            trade_count = int(result.metrics.get("trade_count", 0))
            exposure = float((signal != 0).mean()) if len(signal) else 0.0
            expectancy = float(result.metrics.get("expectancy", 0.0))
            usable = (
                trade_count >= int(conf.min_trades_window)
                and exposure >= float(conf.min_exposure_window)
                and np.isfinite(expectancy)
            )
            if usable:
                usable_windows += 1
            pnl = pd.to_numeric(result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).tolist()
            pooled.extend([float(v) for v in pnl])

        wf_executed = bool(total_windows > 0 and usable_windows >= int(conf.min_usable_windows))
        mc_triggered = bool(wf_executed and len(pooled) >= int(conf.mc_min_trades))
        exp_lcb = float(bootstrap_lcb(values=np.asarray(pooled, dtype=float), seed=int(conf.seed), samples=500)) if pooled else 0.0
        rows.append(
            {
                "candidate_id": str(item.get("candidate_id", "")),
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "usable_windows": int(usable_windows),
                "total_windows": int(total_windows),
                "wf_executed": bool(wf_executed),
                "mc_triggered": bool(mc_triggered),
                "pooled_trades": int(len(pooled)),
                "exp_lcb": float(exp_lcb),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {"wf_executed_pct": 0.0, "mc_trigger_pct": 0.0, "validated_count": 0}
    wf_pct = float(out["wf_executed"].mean() * 100.0)
    mc_pct = float(out["mc_triggered"].mean() * 100.0)
    return out, {
        "wf_executed_pct": wf_pct,
        "mc_trigger_pct": mc_pct,
        "validated_count": int(out.shape[0]),
    }

