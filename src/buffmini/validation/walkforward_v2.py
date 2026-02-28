"""Stage-8.1 standardized multi-window walk-forward evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals
from buffmini.types import StrategySpec


@dataclass(frozen=True)
class WindowTriplet:
    """One train/holdout/forward triplet with non-overlapping windows."""

    window_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    forward_start: pd.Timestamp
    forward_end: pd.Timestamp


def build_windows(
    start_ts: str | pd.Timestamp,
    end_ts: str | pd.Timestamp,
    train_days: int,
    holdout_days: int,
    forward_days: int,
    step_days: int,
    reserve_tail_days: int = 0,
) -> list[WindowTriplet]:
    """Build deterministic UTC walk-forward windows.

    Window semantics:
    - train: [train_start, train_end)
    - holdout: [holdout_start, holdout_end)
    - forward: [forward_start, forward_end)
    """

    start = _ensure_utc(start_ts)
    end = _ensure_utc(end_ts)
    if end <= start:
        raise ValueError("end_ts must be after start_ts")
    if min(int(train_days), int(holdout_days), int(forward_days), int(step_days)) < 1:
        raise ValueError("train_days, holdout_days, forward_days, step_days must all be >= 1")
    if int(reserve_tail_days) < 0:
        raise ValueError("reserve_tail_days must be >= 0")

    train_delta = pd.Timedelta(days=int(train_days))
    holdout_delta = pd.Timedelta(days=int(holdout_days))
    forward_delta = pd.Timedelta(days=int(forward_days))
    step_effective_days = max(int(step_days), int(holdout_days) + int(forward_days))
    step_delta = pd.Timedelta(days=step_effective_days)
    reserve_delta = pd.Timedelta(days=int(reserve_tail_days))
    train_end_cutoff = end - reserve_delta

    windows: list[WindowTriplet] = []
    train_end = start + train_delta
    idx = 0
    while True:
        train_start = train_end - train_delta
        holdout_start = train_end
        holdout_end = holdout_start + holdout_delta
        forward_start = holdout_end
        forward_end = forward_start + forward_delta

        if train_end > train_end_cutoff:
            break
        if forward_end > end:
            break

        windows.append(
            WindowTriplet(
                window_idx=idx,
                train_start=train_start,
                train_end=train_end,
                holdout_start=holdout_start,
                holdout_end=holdout_end,
                forward_start=forward_start,
                forward_end=forward_end,
            )
        )
        idx += 1
        train_end += step_delta

    return windows


def evaluate_candidate_on_window(
    candidate: dict[str, Any],
    data: pd.DataFrame,
    window_triplet: WindowTriplet,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one candidate on one window triplet without future leakage."""

    strategy = candidate.get("strategy")
    if not isinstance(strategy, StrategySpec):
        raise ValueError("candidate['strategy'] must be StrategySpec")
    symbol = str(candidate.get("symbol", "BTC/USDT"))
    signal_col = str(candidate.get("signal_col", "signal"))
    gating_mode = str(candidate.get("gating_mode", "none"))

    frame = data.copy().sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    train = _slice_window(frame, window_triplet.train_start, window_triplet.train_end)
    holdout = _slice_window(frame, window_triplet.holdout_start, window_triplet.holdout_end)
    forward = _slice_window(frame, window_triplet.forward_start, window_triplet.forward_end)

    settings = _walkforward_settings(cfg)
    costs = cfg.get("costs", {})
    risk = cfg.get("risk", {})

    if signal_col not in holdout.columns:
        holdout[signal_col] = generate_signals(holdout, strategy=strategy, gating_mode=gating_mode)
    if signal_col not in forward.columns:
        forward[signal_col] = generate_signals(forward, strategy=strategy, gating_mode=gating_mode)

    holdout_result = _safe_backtest(
        frame=holdout,
        strategy=strategy,
        symbol=symbol,
        signal_col=signal_col,
        cfg=cfg,
        costs=costs,
        risk=risk,
    )
    forward_result = _safe_backtest(
        frame=forward,
        strategy=strategy,
        symbol=symbol,
        signal_col=signal_col,
        cfg=cfg,
        costs=costs,
        risk=risk,
    )

    forward_metrics = _result_metrics(forward_result, forward)
    holdout_metrics = _result_metrics(holdout_result, holdout)
    usable, reasons = _usable_window(
        forward_metrics=forward_metrics,
        min_trades=float(settings["min_trades"]),
        min_exposure=float(settings["min_exposure"]),
    )

    return {
        "window_idx": int(window_triplet.window_idx),
        "train_start": window_triplet.train_start.isoformat(),
        "train_end": window_triplet.train_end.isoformat(),
        "holdout_start": window_triplet.holdout_start.isoformat(),
        "holdout_end": window_triplet.holdout_end.isoformat(),
        "forward_start": window_triplet.forward_start.isoformat(),
        "forward_end": window_triplet.forward_end.isoformat(),
        "train_bars": int(len(train)),
        "holdout_bars": int(len(holdout)),
        "forward_bars": int(len(forward)),
        "usable": bool(usable),
        "exclude_reasons": ";".join(reasons),
        "holdout_expectancy": float(holdout_metrics["expectancy"]),
        "holdout_profit_factor": float(holdout_metrics["profit_factor"]),
        "holdout_max_drawdown": float(holdout_metrics["max_drawdown"]),
        "holdout_return_pct": float(holdout_metrics["return_pct"]),
        "holdout_trade_count": int(holdout_metrics["trade_count"]),
        "holdout_exposure_ratio": float(holdout_metrics["exposure_ratio"]),
        "forward_expectancy": float(forward_metrics["expectancy"]),
        "forward_profit_factor": float(forward_metrics["profit_factor"]),
        "forward_max_drawdown": float(forward_metrics["max_drawdown"]),
        "forward_return_pct": float(forward_metrics["return_pct"]),
        "forward_trade_count": int(forward_metrics["trade_count"]),
        "forward_exposure_ratio": float(forward_metrics["exposure_ratio"]),
    }


def aggregate_windows(metrics_per_window: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Aggregate per-window metrics into robust summary and classification."""

    settings = _walkforward_settings(cfg)
    rows = sorted(metrics_per_window, key=lambda row: int(row.get("window_idx", 0)))
    usable_rows = [row for row in rows if bool(row.get("usable", False))]
    excluded_rows = [row for row in rows if not bool(row.get("usable", False))]

    reason_hist: dict[str, int] = {}
    for row in excluded_rows:
        reasons = [item.strip() for item in str(row.get("exclude_reasons", "")).split(";") if item.strip()]
        if not reasons:
            reasons = ["unspecified"]
        for reason in reasons:
            reason_hist[reason] = reason_hist.get(reason, 0) + 1

    forward_expectancy = _series(usable_rows, "forward_expectancy")
    forward_profit_factor = _series(usable_rows, "forward_profit_factor")
    forward_return = _series(usable_rows, "forward_return_pct")
    forward_max_dd = _series(usable_rows, "forward_max_drawdown")

    summary = {
        "total_windows": int(len(rows)),
        "usable_windows": int(len(usable_rows)),
        "excluded_windows": int(len(excluded_rows)),
        "excluded_reasons": dict(sorted(reason_hist.items())),
        "forward_expectancy": _robust_stats(forward_expectancy),
        "forward_profit_factor": _robust_stats(forward_profit_factor),
        "forward_return_pct": _robust_stats(forward_return),
        "forward_max_drawdown": _robust_stats(forward_max_dd),
    }

    min_usable_windows = int(settings["min_usable_windows"])
    if len(usable_rows) < min_usable_windows:
        classification = "INSUFFICIENT_DATA"
        explanation = f"usable_windows={len(usable_rows)} < min_usable_windows={min_usable_windows}"
    else:
        stable = (
            summary["forward_expectancy"]["median"] >= float(settings["stable_min_median_expectancy"])
            and summary["forward_expectancy"]["worst"] >= float(settings["stable_min_worst_expectancy"])
            and summary["forward_profit_factor"]["median"] >= float(settings["stable_min_median_profit_factor"])
            and summary["forward_return_pct"]["p05"] >= float(settings["stable_min_p05_return"])
            and summary["forward_max_drawdown"]["median"] <= float(settings["stable_max_median_max_drawdown"])
        )
        classification = "STABLE" if stable else "UNSTABLE"
        explanation = "all robust stability thresholds passed" if stable else "one or more robustness thresholds failed"

    summary["classification"] = classification
    summary["classification_explanation"] = explanation
    return summary


def write_walkforward_v2_artifacts(
    run_dir: Path,
    window_metrics: list[dict[str, Any]],
    summary: dict[str, Any],
    command: str | None = None,
) -> Path:
    """Write Stage-8.1 artifacts under one run directory."""

    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(sorted(window_metrics, key=lambda row: int(row.get("window_idx", 0))))
    metrics_df.to_csv(out_dir / "window_metrics_full.csv", index=False)
    excluded_df = metrics_df.loc[metrics_df["usable"] == False].copy() if not metrics_df.empty else pd.DataFrame()  # noqa: E712
    excluded_df.to_csv(out_dir / "excluded_windows.csv", index=False)

    payload = dict(summary)
    payload["command"] = command
    (out_dir / "walkforward_v2_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# Walk-Forward V2 Report")
    lines.append("")
    if command:
        lines.append(f"- command: `{command}`")
    lines.append(f"- classification: `{summary['classification']}`")
    lines.append(f"- usable_windows: `{summary['usable_windows']}` / `{summary['total_windows']}`")
    lines.append(f"- excluded_reasons: `{summary['excluded_reasons']}`")
    lines.append(f"- explanation: {summary['classification_explanation']}")
    lines.append("")
    lines.append("## Robust Metrics (Forward)")
    lines.append("| metric | median | iqr | p05 | worst |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for metric_key in ["forward_expectancy", "forward_profit_factor", "forward_return_pct", "forward_max_drawdown"]:
        stats = summary[metric_key]
        lines.append(
            f"| {metric_key} | {float(stats['median']):.6f} | {float(stats['iqr']):.6f} | "
            f"{float(stats['p05']):.6f} | {float(stats['worst']):.6f} |"
        )
    lines.append("")
    lines.append("## Per-Window Table")
    if metrics_df.empty:
        lines.append("- no windows generated")
    else:
        lines.append(
            "| idx | usable | forward_expectancy | forward_pf | forward_max_dd | forward_return | trade_count | exposure | reasons |"
        )
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in metrics_df.to_dict(orient="records"):
            lines.append(
                f"| {int(row['window_idx'])} | {bool(row['usable'])} | {float(row['forward_expectancy']):.6f} | "
                f"{float(row['forward_profit_factor']):.6f} | {float(row['forward_max_drawdown']):.6f} | "
                f"{float(row['forward_return_pct']):.6f} | {int(row['forward_trade_count'])} | "
                f"{float(row['forward_exposure_ratio']):.6f} | {row.get('exclude_reasons', '')} |"
            )

    (out_dir / "walkforward_v2_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return out_dir


def _safe_backtest(
    frame: pd.DataFrame,
    strategy: StrategySpec,
    symbol: str,
    signal_col: str,
    cfg: dict[str, Any],
    costs: dict[str, Any],
    risk: dict[str, Any],
) -> Any:
    if frame.empty:
        return None
    settings = _walkforward_settings(cfg)
    return run_backtest(
        frame=frame,
        strategy_name=strategy.name,
        symbol=symbol,
        signal_col=signal_col,
        initial_capital=float(settings["initial_capital"] or (10_000.0 * float(risk.get("max_concurrent_positions", 1)))),
        stop_atr_multiple=float(settings["stop_atr_multiple"]),
        take_profit_atr_multiple=float(settings["take_profit_atr_multiple"]),
        max_hold_bars=int(settings["max_hold_bars"]),
        round_trip_cost_pct=float(costs.get("round_trip_cost_pct", 0.1)),
        slippage_pct=float(costs.get("slippage_pct", 0.0005)),
        cost_model_cfg=cfg.get("cost_model"),
    )


def _result_metrics(result: Any, frame: pd.DataFrame) -> dict[str, float]:
    if result is None or frame.empty:
        return {
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "return_pct": 0.0,
            "trade_count": 0.0,
            "exposure_ratio": 0.0,
            "finite": True,
        }
    trade_count = float(result.metrics.get("trade_count", 0.0))
    bars = max(1, len(frame))
    exposure_ratio = 0.0
    if not result.trades.empty and "bars_held" in result.trades.columns:
        exposure_ratio = float(result.trades["bars_held"].astype(float).sum() / float(bars))
    final_equity = float(result.equity_curve["equity"].iloc[-1]) if not result.equity_curve.empty else 0.0
    initial_equity = float(result.equity_curve["equity"].iloc[0]) if not result.equity_curve.empty else 1.0
    return_pct = float((final_equity / initial_equity) - 1.0) if initial_equity != 0 else 0.0
    metrics = {
        "expectancy": float(result.metrics.get("expectancy", 0.0)),
        "profit_factor": float(result.metrics.get("profit_factor", 0.0)),
        "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
        "return_pct": return_pct,
        "trade_count": trade_count,
        "exposure_ratio": exposure_ratio,
    }
    metrics["finite"] = _finite_metrics(metrics)
    return metrics


def _usable_window(forward_metrics: dict[str, float], min_trades: float, min_exposure: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not bool(forward_metrics.get("finite", False)):
        reasons.append("non_finite_metrics")
    if float(forward_metrics["trade_count"]) < float(min_trades):
        reasons.append("min_trades")
    if float(forward_metrics["exposure_ratio"]) < float(min_exposure):
        reasons.append("min_exposure")
    return len(reasons) == 0, reasons


def _walkforward_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "min_trades": 10,
        "min_exposure": 0.01,
        "min_usable_windows": 3,
        "stable_min_median_expectancy": 0.0,
        "stable_min_worst_expectancy": 0.0,
        "stable_min_median_profit_factor": 1.0,
        "stable_min_p05_return": 0.0,
        "stable_max_median_max_drawdown": 0.25,
        "stop_atr_multiple": 1.5,
        "take_profit_atr_multiple": 3.0,
        "max_hold_bars": 24,
        "initial_capital": None,
    }
    overrides = (
        cfg.get("evaluation", {})
        .get("stage8", {})
        .get("walkforward_v2", {})
        if isinstance(cfg, dict)
        else {}
    )
    resolved = dict(defaults)
    if isinstance(overrides, dict):
        resolved.update(overrides)
    return resolved


def _series(rows: list[dict[str, Any]], key: str) -> pd.Series:
    if not rows:
        return pd.Series(dtype=float)
    values = [float(row.get(key, 0.0)) for row in rows]
    return pd.Series(values, dtype=float)


def _robust_stats(values: pd.Series) -> dict[str, float]:
    if values.empty:
        return {"median": 0.0, "iqr": 0.0, "p05": 0.0, "worst": 0.0}
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"median": 0.0, "iqr": 0.0, "p05": 0.0, "worst": 0.0}
    q75 = float(clean.quantile(0.75))
    q25 = float(clean.quantile(0.25))
    return {
        "median": float(clean.median()),
        "iqr": float(q75 - q25),
        "p05": float(clean.quantile(0.05)),
        "worst": float(clean.min()),
    }


def _finite_metrics(metrics: dict[str, float]) -> bool:
    for value in metrics.values():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def _slice_window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    mask = (ts >= _ensure_utc(start)) & (ts < _ensure_utc(end))
    return frame.loc[mask].copy().reset_index(drop=True)


def _ensure_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
