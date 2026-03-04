"""Context-first candidate discovery and evaluation for Stage-28."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.stage10.exits import normalize_exit_mode
from buffmini.stage26.conditional_eval import bootstrap_lcb
from buffmini.stage26.rulelets import build_rulelet_library
from buffmini.utils.hashing import stable_hash


@dataclass(frozen=True)
class ContextCandidate:
    name: str
    family: str
    context: str
    threshold: float
    default_exit: str
    required_features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "context": self.context,
            "threshold": float(self.threshold),
            "default_exit": self.default_exit,
            "required_features": list(self.required_features),
        }


def discover_candidates_for_context(
    *,
    context: str,
    symbol: str,
    timeframe: str,
    rulelet_library: dict[str, Any] | None = None,
) -> list[ContextCandidate]:
    """Discover contextual candidates for one context label."""

    _ = (symbol, timeframe)  # Kept for API stability and future symbol-aware filtering.
    context_name = str(context)
    lib = rulelet_library or build_rulelet_library()
    candidates: list[ContextCandidate] = []
    for name, rulelet in sorted(lib.items()):
        allowed = tuple(str(item) for item in getattr(rulelet, "contexts_allowed", tuple()))
        if context_name not in allowed:
            continue
        required = tuple(str(item) for item in rulelet.required_features())
        candidates.append(
            ContextCandidate(
                name=str(name),
                family=str(getattr(rulelet, "family", "unknown")),
                context=context_name,
                threshold=float(getattr(rulelet, "threshold", 0.30)),
                default_exit=str(getattr(rulelet, "default_exit", "fixed_atr")),
                required_features=required,
            )
        )
    return candidates


def compute_context_signal(
    *,
    frame: pd.DataFrame,
    candidate: ContextCandidate,
    rulelet_library: dict[str, Any] | None = None,
    shift_entries: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute candidate score/signal restricted to the candidate context."""

    lib = rulelet_library or build_rulelet_library()
    rulelet = lib[str(candidate.name)]
    ctx = frame.get("ctx_state", pd.Series("RANGE", index=frame.index)).astype(str)
    mask = (ctx == str(candidate.context)).astype(bool)
    score = pd.to_numeric(rulelet.compute_score(frame), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    signal = pd.Series(0, index=frame.index, dtype=int)
    signal.loc[mask & (score >= float(candidate.threshold))] = 1
    signal.loc[mask & (score <= -float(candidate.threshold))] = -1
    if bool(shift_entries):
        signal = signal.shift(1).fillna(0).astype(int)
    return score, signal, mask


def evaluate_candidate_in_context(
    *,
    frame: pd.DataFrame,
    candidate: ContextCandidate,
    symbol: str,
    timeframe: str,
    seed: int,
    cost_levels: list[dict[str, Any]] | None = None,
    rulelet_library: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate only on its active context mask."""

    lib = rulelet_library or build_rulelet_library()
    frame_eval = frame.copy()
    if "atr_14" not in frame_eval.columns:
        high = pd.to_numeric(frame_eval.get("high"), errors="coerce")
        low = pd.to_numeric(frame_eval.get("low"), errors="coerce")
        close = pd.to_numeric(frame_eval.get("close"), errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        frame_eval["atr_14"] = tr.rolling(14, min_periods=2).mean().fillna(0.0)

    _, signal, mask = compute_context_signal(
        frame=frame_eval,
        candidate=candidate,
        rulelet_library=lib,
        shift_entries=True,
    )
    ts = pd.to_datetime(frame_eval.get("timestamp"), utc=True, errors="coerce")
    occurrences = int(mask.sum())
    costs = cost_levels or [
        {
            "name": "realistic",
            "round_trip_cost_pct": 0.1,
            "slippage_pct": 0.0005,
            "cost_model_cfg": {},
            "stop_atr_multiple": 1.5,
            "take_profit_atr_multiple": 3.0,
            "max_hold_bars": 24,
        }
    ]
    rows: list[dict[str, Any]] = []
    for cost in costs:
        result = run_backtest(
            frame=frame_eval.assign(signal=signal),
            strategy_name=f"Stage28::{candidate.name}",
            symbol=str(symbol),
            signal_col="signal",
            stop_atr_multiple=float(cost.get("stop_atr_multiple", 1.5)),
            take_profit_atr_multiple=float(cost.get("take_profit_atr_multiple", 3.0)),
            max_hold_bars=int(cost.get("max_hold_bars", 24)),
            round_trip_cost_pct=float(cost.get("round_trip_cost_pct", 0.1)),
            slippage_pct=float(cost.get("slippage_pct", 0.0005)),
            exit_mode=normalize_exit_mode(str(candidate.default_exit)),
            cost_model_cfg=cost.get("cost_model_cfg", {}),
        )
        trades = result.trades.copy()
        if "entry_time" in trades.columns and not trades.empty:
            entry_time = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
            trades_ctx = trades.loc[entry_time.isin(ts.loc[mask])].copy()
        else:
            trades_ctx = trades.iloc[0:0].copy()
        pnl = pd.to_numeric(trades_ctx.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        exp_lcb = bootstrap_lcb(
            values=pnl,
            seed=int(stable_seed(seed, symbol, timeframe, candidate.name, candidate.context, str(cost.get("name", "cost")))),
            samples=300,
        )
        rows.append(
            {
                "cost_level": str(cost.get("name", "cost")),
                "trades": int(trades_ctx.shape[0]),
                "expectancy": float(np.mean(pnl)) if pnl.size else 0.0,
                "exp_lcb": float(exp_lcb),
                "profit_factor": float(result.metrics.get("profit_factor", 0.0)),
                "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
            }
        )

    trade_counts = np.asarray([float(item["trades"]) for item in rows], dtype=float)
    expectancies = np.asarray([float(item["expectancy"]) for item in rows], dtype=float)
    lcbs = np.asarray([float(item["exp_lcb"]) for item in rows], dtype=float)
    pfs = np.asarray([float(item["profit_factor"]) for item in rows], dtype=float)
    maxdds = np.asarray([float(item["max_drawdown"]) for item in rows], dtype=float)
    trade_median = int(np.median(trade_counts)) if trade_counts.size else 0
    expectancy = float(np.median(expectancies)) if expectancies.size else 0.0
    exp_lcb = float(np.median(lcbs)) if lcbs.size else 0.0
    pf = float(np.median(pfs)) if pfs.size else 0.0
    maxdd = float(np.median(maxdds)) if maxdds.size else 0.0
    cost_sensitivity = float(np.max(expectancies) - np.min(expectancies)) if expectancies.size else 0.0
    if trade_median < 10 and exp_lcb > 0.0:
        classification = "RARE"
    elif exp_lcb > 0.0:
        classification = "PASS"
    elif trade_median > 0 and expectancy > 0.0:
        classification = "WEAK"
    else:
        classification = "FAIL"
    return {
        "symbol": str(symbol),
        "timeframe": str(timeframe),
        "candidate": candidate.to_dict(),
        "context": str(candidate.context),
        "context_occurrences": int(occurrences),
        "trades_in_context": int(trade_median),
        "expectancy": float(expectancy),
        "exp_lcb": float(exp_lcb),
        "profit_factor": float(pf),
        "max_drawdown": float(maxdd),
        "cost_sensitivity": float(cost_sensitivity),
        "classification": str(classification),
        "cost_rows": rows,
    }


def evaluate_context_candidate_matrix(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    seed: int,
    contexts: list[str] | None = None,
    cost_levels: list[dict[str, Any]] | None = None,
    rulelet_library: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate all discovered candidates per context and return a matrix."""

    lib = rulelet_library or build_rulelet_library()
    if contexts is None:
        ctx_col = frame.get("ctx_state", pd.Series(dtype=str)).astype(str)
        contexts = sorted(set(ctx_col.tolist()))
    rows: list[dict[str, Any]] = []
    for context in contexts:
        candidates = discover_candidates_for_context(
            context=str(context),
            symbol=str(symbol),
            timeframe=str(timeframe),
            rulelet_library=lib,
        )
        for candidate in candidates:
            result = evaluate_candidate_in_context(
                frame=frame,
                candidate=candidate,
                symbol=str(symbol),
                timeframe=str(timeframe),
                seed=int(seed),
                cost_levels=cost_levels,
                rulelet_library=lib,
            )
            candidate_id = stable_hash(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candidate": candidate.name,
                    "context": context,
                },
                length=12,
            )
            rows.append(
                {
                    "symbol": str(symbol),
                    "timeframe": str(timeframe),
                    "candidate_id": str(candidate_id),
                    "candidate": str(candidate.name),
                    "family": str(candidate.family),
                    "context": str(context),
                    "context_occurrences": int(result["context_occurrences"]),
                    "trades_in_context": int(result["trades_in_context"]),
                    "expectancy": float(result["expectancy"]),
                    "exp_lcb": float(result["exp_lcb"]),
                    "profit_factor": float(result["profit_factor"]),
                    "max_drawdown": float(result["max_drawdown"]),
                    "cost_sensitivity": float(result["cost_sensitivity"]),
                    "classification": str(result["classification"]),
                }
            )
    return pd.DataFrame(rows)


def stable_seed(seed: int, *parts: object) -> int:
    text = "|".join([str(seed), *[str(part) for part in parts]])
    return int(stable_hash(text, length=8), 16) % (2**31 - 1)
