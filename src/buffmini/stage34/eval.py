"""Stage-34 strict evaluation (rolling WF + MC + cost stress)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.stage26.conditional_eval import bootstrap_lcb
from buffmini.stage28.window_calendar import generate_window_calendar
from buffmini.stage34.train import predict_model_proba


@dataclass(frozen=True)
class EvalConfig:
    threshold: float = 0.55
    window_months: tuple[int, ...] = (3, 6)
    step_months: int = 1
    min_window_trades: int = 8
    min_window_exposure: float = 0.01
    min_usable_windows: int = 3
    mc_min_trades: int = 30
    seed: int = 42
    costs: tuple[dict[str, Any], ...] = (
        {"name": "research", "round_trip_cost_pct": 0.05, "slippage_pct": 0.0003},
        {"name": "live", "round_trip_cost_pct": 0.10, "slippage_pct": 0.0005},
    )


def evaluate_models_strict(
    dataset: pd.DataFrame,
    *,
    models: dict[str, dict[str, Any]],
    cfg: EvalConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if dataset.empty:
        return pd.DataFrame(), {"status": "EMPTY_DATASET", "wf_executed_pct": 0.0, "mc_trigger_pct": 0.0}
    work = dataset.copy().sort_values("timestamp").reset_index(drop=True)
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    rows: list[dict[str, Any]] = []

    for model_name, model in sorted(models.items()):
        proba = predict_model_proba(model, work)
        work_model = work.copy()
        work_model["proba"] = np.clip(np.asarray(proba, dtype=float), 1e-6, 1.0 - 1e-6)
        work_model["score"] = 2.0 * work_model["proba"] - 1.0
        signal = pd.Series(0, index=work_model.index, dtype=int)
        thr = float(cfg.threshold)
        signal.loc[work_model["proba"] >= thr] = 1
        signal.loc[work_model["proba"] <= (1.0 - thr)] = -1
        work_model["signal"] = signal.shift(1).fillna(0).astype(int)
        for window_m in [int(v) for v in cfg.window_months]:
            calendar = generate_window_calendar(
                work_model["timestamp"],
                window_months=int(window_m),
                step_months=int(cfg.step_months),
            )
            for cost in [dict(v) for v in cfg.costs]:
                row = _evaluate_one(
                    frame=work_model,
                    model_name=str(model_name),
                    threshold=float(cfg.threshold),
                    window_months=int(window_m),
                    calendar=calendar,
                    cost=cost,
                    cfg=cfg,
                )
                rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {"status": "NO_ROWS", "wf_executed_pct": 0.0, "mc_trigger_pct": 0.0}
    summary = {
        "status": "OK",
        "rows": int(out.shape[0]),
        "wf_executed_pct": float(out["wf_executed"].mean() * 100.0),
        "mc_trigger_pct": float(out["mc_triggered"].mean() * 100.0),
        "top_model_by_live_exp_lcb": str(
            out.loc[out["cost_mode"] == "live"].sort_values("exp_lcb", ascending=False).head(1)["model_name"].iloc[0]
            if not out.loc[out["cost_mode"] == "live"].empty
            else out.sort_values("exp_lcb", ascending=False).head(1)["model_name"].iloc[0]
        ),
        "final_verdict": _final_verdict(out),
        "failure_mode_counts": out["failure_mode"].value_counts().to_dict(),
    }
    return out, summary


def _evaluate_one(
    *,
    frame: pd.DataFrame,
    model_name: str,
    threshold: float,
    window_months: int,
    calendar: pd.DataFrame,
    cost: dict[str, Any],
    cfg: EvalConfig,
) -> dict[str, Any]:
    if calendar.empty:
        return {
            "model_name": model_name,
            "cost_mode": str(cost.get("name", "")),
            "threshold": float(threshold),
            "window_months": int(window_months),
            "windows_total": 0,
            "usable_windows": 0,
            "wf_executed": False,
            "mc_triggered": False,
            "trade_count": 0,
            "exp_lcb": 0.0,
            "expectancy": 0.0,
            "pf_adj": 0.0,
            "maxdd_p95": 0.0,
            "positive_windows_ratio": 0.0,
            "failure_mode": "insufficient_sample",
        }
    pooled: list[float] = []
    usable = 0
    positives = 0
    trade_count = 0
    exp_rows: list[float] = []
    pf_rows: list[float] = []
    dd_rows: list[float] = []
    for rec in calendar.to_dict(orient="records"):
        start = pd.to_datetime(rec["window_start"], utc=True, errors="coerce")
        end = pd.to_datetime(rec["window_end"], utc=True, errors="coerce")
        local = frame.loc[(frame["timestamp"] >= start) & (frame["timestamp"] < end)].reset_index(drop=True)
        if local.shape[0] < 100:
            continue
        result = run_backtest(
            frame=local,
            strategy_name=f"Stage34::{model_name}",
            symbol=str(local.get("symbol", pd.Series(["BTC/USDT"])).iloc[0] if "symbol" in local.columns else "BTC/USDT"),
            signal_col="signal",
            stop_atr_multiple=1.5,
            take_profit_atr_multiple=3.0,
            max_hold_bars=24,
            round_trip_cost_pct=float(cost.get("round_trip_cost_pct", 0.1)),
            slippage_pct=float(cost.get("slippage_pct", 0.0005)),
            exit_mode="fixed_atr",
            cost_model_cfg={},
        )
        trades = result.trades.copy()
        pnl = pd.to_numeric(trades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        local_trades = int(trades.shape[0])
        local_exp = float(result.metrics.get("expectancy", 0.0))
        local_pf = float(np.clip(float(result.metrics.get("profit_factor", 0.0)), 0.0, 10.0))
        local_dd = float(result.metrics.get("max_drawdown", 0.0))
        exposure = float((pd.to_numeric(local.get("signal", 0), errors="coerce") != 0).mean())
        if local_trades >= int(cfg.min_window_trades) and exposure >= float(cfg.min_window_exposure) and np.isfinite(local_exp):
            usable += 1
            if local_exp > 0.0:
                positives += 1
            pooled.extend([float(v) for v in pnl.to_list()])
            trade_count += int(local_trades)
            exp_rows.append(local_exp)
            pf_rows.append(local_pf)
            dd_rows.append(local_dd)

    wf_executed = bool(usable >= int(cfg.min_usable_windows))
    mc_triggered = bool(wf_executed and trade_count >= int(cfg.mc_min_trades))
    exp_lcb = float(bootstrap_lcb(values=np.asarray(pooled, dtype=float), seed=int(cfg.seed), samples=500)) if pooled else 0.0
    expectancy = float(np.mean(exp_rows)) if exp_rows else 0.0
    pf_adj = float(np.mean(pf_rows)) if pf_rows else 0.0
    maxdd = float(np.quantile(np.asarray(dd_rows, dtype=float), 0.95)) if dd_rows else 0.0
    pos_ratio = float(positives / max(1, usable))
    failure_mode = classify_failure_mode(
        wf_executed=wf_executed,
        mc_triggered=mc_triggered,
        trade_count=int(trade_count),
        exp_lcb=float(exp_lcb),
        threshold=float(threshold),
    )
    return {
        "model_name": model_name,
        "cost_mode": str(cost.get("name", "")),
        "threshold": float(threshold),
        "window_months": int(window_months),
        "windows_total": int(calendar.shape[0]),
        "usable_windows": int(usable),
        "wf_executed": bool(wf_executed),
        "mc_triggered": bool(mc_triggered),
        "trade_count": int(trade_count),
        "exp_lcb": float(exp_lcb),
        "expectancy": float(expectancy),
        "pf_adj": float(pf_adj),
        "maxdd_p95": float(maxdd),
        "positive_windows_ratio": float(pos_ratio),
        "failure_mode": str(failure_mode),
    }


def classify_failure_mode(*, wf_executed: bool, mc_triggered: bool, trade_count: int, exp_lcb: float, threshold: float) -> str:
    if int(trade_count) <= 0:
        return "no_trades_due_to_thresholds" if float(threshold) > 0.5 else "policy_zero_activation"
    if not bool(wf_executed):
        return "insufficient_sample"
    if bool(wf_executed) and not bool(mc_triggered):
        return "unstable_edge"
    if float(exp_lcb) <= 0.0:
        return "no_trades_due_to_cost_drag"
    return "ok"


def _final_verdict(rows: pd.DataFrame) -> str:
    live = rows.loc[rows["cost_mode"] == "live"].copy()
    target = live if not live.empty else rows
    if target.empty:
        return "INSUFFICIENT_DATA"
    robust = target.loc[(target["wf_executed"]) & (target["mc_triggered"]) & (target["exp_lcb"] > 0.0)]
    if robust.empty:
        if bool(target["wf_executed"].any()):
            return "NO_EDGE"
        return "INSUFFICIENT_DATA"
    strong = robust.loc[(robust["positive_windows_ratio"] >= 0.60) & (robust["trade_count"] >= 100)]
    if not strong.empty:
        return "EDGE"
    return "WEAK_EDGE"
