"""Stage-26 conditional evaluation with bootstrap LCB and rolling slices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.stage10.exits import normalize_exit_mode
from buffmini.utils.hashing import stable_hash


@dataclass(frozen=True)
class ConditionalEvalParams:
    bootstrap_samples: int = 500
    seed: int = 42
    min_occurrences: int = 30
    min_trades: int = 30
    rare_min_trades: int = 10
    rolling_months: tuple[int, ...] = (3, 6, 12)


def evaluate_rulelets_conditionally(
    *,
    frame: pd.DataFrame,
    rulelets: dict[str, Any],
    symbol: str,
    timeframe: str,
    seed: int,
    cost_levels: list[dict[str, Any]],
    params: ConditionalEvalParams | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate each rulelet inside allowed contexts."""

    p = params or ConditionalEvalParams(seed=int(seed))
    out_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"symbol": str(symbol), "timeframe": str(timeframe), "rows": []}
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
    ctx = frame.get("ctx_state", pd.Series("RANGE", index=frame.index)).astype(str)

    for rulelet_name, rulelet in rulelets.items():
        score = pd.to_numeric(rulelet.compute_score(frame), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        threshold = float(getattr(rulelet, "threshold", 0.30))
        allowed = set(getattr(rulelet, "contexts_allowed", tuple()))
        for context in sorted(allowed):
            mask = ctx == str(context)
            occurrences = int(mask.sum())
            signal = pd.Series(0, index=frame.index, dtype=int)
            active_score = score.where(mask, 0.0)
            signal.loc[active_score >= threshold] = 1
            signal.loc[active_score <= -threshold] = -1
            signal = signal.shift(1).fillna(0).astype(int)

            all_cost_rows: list[dict[str, Any]] = []
            for cost in cost_levels:
                cfg = dict(cost)
                result = run_backtest(
                    frame=frame.assign(signal=signal),
                    strategy_name=f"Stage26::{rulelet_name}",
                    symbol=str(symbol),
                    signal_col="signal",
                    stop_atr_multiple=float(cfg.get("stop_atr_multiple", 1.5)),
                    take_profit_atr_multiple=float(cfg.get("take_profit_atr_multiple", 3.0)),
                    max_hold_bars=int(cfg.get("max_hold_bars", 24)),
                    round_trip_cost_pct=float(cfg.get("round_trip_cost_pct", 0.1)),
                    slippage_pct=float(cfg.get("slippage_pct", 0.0005)),
                    exit_mode=normalize_exit_mode(str(getattr(rulelet, "default_exit", "fixed_atr"))),
                    cost_model_cfg=cfg.get("cost_model_cfg", {}),
                )
                trades = result.trades.copy()
                if "entry_time" in trades.columns and not trades.empty:
                    entry_time = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
                    trades_context = trades.loc[entry_time.isin(ts.loc[mask])].copy()
                else:
                    trades_context = trades.iloc[0:0].copy()
                pnl = pd.to_numeric(trades_context.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
                exp_lcb = bootstrap_lcb(values=pnl, seed=_seed_for(seed, symbol, timeframe, rulelet_name, context, str(cfg.get("name", "cost"))), samples=int(p.bootstrap_samples))
                rolling = _rolling_context_metrics(
                    ts=ts,
                    signal=signal,
                    ctx_mask=mask,
                    months=tuple(p.rolling_months),
                )
                all_cost_rows.append(
                    {
                        "cost_level": str(cfg.get("name", "cost")),
                        "trade_count": int(trades_context.shape[0]),
                        "expectancy": float(np.mean(pnl)) if pnl.size else 0.0,
                        "exp_lcb": float(exp_lcb),
                        "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
                        "rolling": rolling,
                    }
                )

            # Aggregate by median across cost levels.
            trade_counts = np.asarray([float(item["trade_count"]) for item in all_cost_rows], dtype=float)
            expectancies = np.asarray([float(item["expectancy"]) for item in all_cost_rows], dtype=float)
            lcbs = np.asarray([float(item["exp_lcb"]) for item in all_cost_rows], dtype=float)
            maxdds = np.asarray([float(item["max_drawdown"]) for item in all_cost_rows], dtype=float)
            trades_in_context = int(np.median(trade_counts)) if trade_counts.size else 0
            expectancy = float(np.median(expectancies)) if expectancies.size else 0.0
            exp_lcb = float(np.median(lcbs)) if lcbs.size else 0.0
            maxdd = float(np.median(maxdds)) if maxdds.size else 0.0
            classification = _classify_result(
                occurrences=occurrences,
                trades=trades_in_context,
                exp_lcb=exp_lcb,
                min_occurrences=int(p.min_occurrences),
                min_trades=int(p.min_trades),
                rare_min_trades=int(p.rare_min_trades),
            )
            row = {
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "rulelet": str(rulelet_name),
                "family": str(getattr(rulelet, "family", "unknown")),
                "context": str(context),
                "context_occurrences": int(occurrences),
                "trades_in_context": int(trades_in_context),
                "expectancy": float(expectancy),
                "exp_lcb": float(exp_lcb),
                "max_drawdown": float(maxdd),
                "classification": str(classification),
                "cost_rows": all_cost_rows,
            }
            out_rows.append(row)
            details["rows"].append(row)

    out = pd.DataFrame(out_rows)
    return out, details


def bootstrap_lcb(*, values: np.ndarray, seed: int, samples: int) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return float(arr[0])
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(max(1, samples)), dtype=float)
    for i in range(means.size):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(np.mean(sample))
    return float(np.quantile(means, 0.05))


def _rolling_context_metrics(
    *,
    ts: pd.Series,
    signal: pd.Series,
    ctx_mask: pd.Series,
    months: tuple[int, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if ts.empty:
        return {str(m): {"bars": 0, "signals": 0} for m in months}
    for m in months:
        end = ts.max()
        start = end - pd.Timedelta(days=int(m) * 30)
        mask = (ts >= start) & (ts <= end) & ctx_mask
        sigs = pd.to_numeric(signal.where(mask, 0), errors="coerce").fillna(0).astype(int)
        out[str(m)] = {
            "bars": int(mask.sum()),
            "signals": int((sigs != 0).sum()),
        }
    return out


def _classify_result(
    *,
    occurrences: int,
    trades: int,
    exp_lcb: float,
    min_occurrences: int,
    min_trades: int,
    rare_min_trades: int,
) -> str:
    if occurrences < int(min_occurrences):
        return "FAIL"
    if trades >= int(min_trades):
        return "PASS" if exp_lcb > 0 else "FAIL"
    if trades >= int(rare_min_trades) and exp_lcb > 0:
        return "RARE"
    if trades > 0 and exp_lcb > 0:
        return "WEAK"
    return "FAIL"


def _seed_for(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    hex_part = stable_hash(text, length=8)
    return int(hex_part, 16) % (2**31 - 1)
