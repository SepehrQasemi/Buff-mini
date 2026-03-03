"""Stage-26 conditional policy replay helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.stage26.conditional_eval import bootstrap_lcb
from buffmini.stage26.policy import build_conditional_policy, compose_policy_signal
from buffmini.stage26.rulelets import build_rulelet_library
from buffmini.utils.hashing import stable_hash


@dataclass(frozen=True)
class ReplayBundle:
    metrics_rows: list[dict[str, Any]]
    policy_trace: pd.DataFrame
    shadow_live_rows: list[dict[str, Any]]
    policy: dict[str, Any]


def replay_conditional_policy(
    *,
    frames_by_symbol_tf: dict[tuple[str, str], pd.DataFrame],
    effects: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    mode: str,
) -> ReplayBundle:
    """Replay composed conditional policy on supplied frames."""

    stage26 = dict((config.get("evaluation", {}) or {}).get("stage26", {}))
    policy_cfg = dict(stage26.get("policy", {}))
    cost_cfg = _base_cost_cfg(config)
    rulelets = build_rulelet_library()
    policy = build_conditional_policy(
        effects=effects,
        min_occurrences_per_context=int(policy_cfg.get("min_occurrences_per_context", 30)),
        min_trades_in_context=int(policy_cfg.get("min_trades_in_context", 30)),
        top_k=int(policy_cfg.get("top_k", 2)),
        w_min=float(policy_cfg.get("w_min", 0.05)),
        w_max=float(policy_cfg.get("w_max", 0.80)),
    )

    metrics_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    shadow_live_rows: list[dict[str, Any]] = []
    for (symbol, timeframe), frame in sorted(frames_by_symbol_tf.items()):
        rulelet_scores = {name: rulelet.compute_score(frame) for name, rulelet in rulelets.items()}
        signal, trace = compose_policy_signal(
            frame=frame,
            rulelet_scores=rulelet_scores,
            policy=policy,
            conflict_mode=str(policy_cfg.get("conflict_mode", "net")),
        )
        result = run_backtest(
            frame=frame.assign(signal=signal),
            strategy_name="Stage26Policy",
            symbol=str(symbol),
            signal_col="signal",
            stop_atr_multiple=float(cost_cfg.get("stop_atr_multiple", 1.5)),
            take_profit_atr_multiple=float(cost_cfg.get("take_profit_atr_multiple", 3.0)),
            max_hold_bars=int(cost_cfg.get("max_hold_bars", 24)),
            round_trip_cost_pct=float(cost_cfg.get("round_trip_cost_pct", 0.1)),
            slippage_pct=float(cost_cfg.get("slippage_pct", 0.0005)),
            exit_mode="fixed_atr",
            cost_model_cfg=cost_cfg.get("cost_model_cfg", {}),
        )
        trade_count = float(result.metrics.get("trade_count", 0.0))
        pnls = pd.to_numeric(result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        exp_lcb = bootstrap_lcb(values=pnls, seed=int(_seed_for(seed, symbol, timeframe, mode)), samples=500)
        months = _months(frame)
        tpm = float(trade_count / max(1e-9, months))
        pf = float(result.metrics.get("profit_factor", 0.0))
        if not np.isfinite(pf):
            pf = 10.0
        metrics_rows.append(
            {
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "mode": str(mode),
                "trade_count": float(trade_count),
                "tpm": float(tpm),
                "exposure_ratio": float((signal != 0).mean()),
                "PF_raw": float(result.metrics.get("profit_factor", 0.0)),
                "PF_clipped": float(np.clip(pf, 0.0, 10.0)),
                "expectancy": float(result.metrics.get("expectancy", 0.0)),
                "exp_lcb": float(exp_lcb),
                "maxDD": float(result.metrics.get("max_drawdown", 0.0)),
                "policy_hash": stable_hash(policy, length=12),
            }
        )
        tr = trace.copy()
        tr["symbol"] = str(symbol)
        tr["timeframe"] = str(timeframe)
        tr["mode"] = str(mode)
        trace_rows.extend(tr.to_dict(orient="records"))
        if str(mode).lower() == "research":
            shadow_live_rows.extend(_shadow_live_checks(trace=tr, frame=frame, config=config))

    return ReplayBundle(
        metrics_rows=metrics_rows,
        policy_trace=pd.DataFrame(trace_rows),
        shadow_live_rows=shadow_live_rows,
        policy=policy,
    )


def _base_cost_cfg(config: dict[str, Any]) -> dict[str, Any]:
    costs = dict(config.get("costs", {}))
    return {
        "round_trip_cost_pct": float(costs.get("round_trip_cost_pct", 0.1)),
        "slippage_pct": float(costs.get("slippage_pct", 0.0005)),
        "cost_model_cfg": config.get("cost_model", {}),
        "stop_atr_multiple": 1.5,
        "take_profit_atr_multiple": 3.0,
        "max_hold_bars": 24,
    }


def _shadow_live_checks(*, trace: pd.DataFrame, frame: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    constraints = dict((config.get("evaluation", {}) or {}).get("constraints", {}))
    live = dict(constraints.get("live", {}))
    min_notional = float(live.get("min_trade_notional", 10.0))
    qty_step = float(live.get("qty_step", 0.0))
    min_qty = float(live.get("min_trade_qty", 0.0))
    close = pd.to_numeric(frame.get("close"), errors="coerce").fillna(0.0)
    out: list[dict[str, Any]] = []
    for idx, row in trace.iterrows():
        sig = int(row.get("final_signal", 0))
        if sig == 0:
            continue
        net = abs(float(row.get("net_score", 0.0)))
        qty = float(max(0.0, net))
        notional = float(qty * close.iloc[idx]) if idx < len(close) else 0.0
        reason = "VALID"
        if qty <= 0.0:
            reason = "SIZE_ZERO"
        elif qty_step > 0.0 and abs((qty / qty_step) - round(qty / qty_step)) > 1e-9:
            reason = "SIZE_TOO_SMALL"
        elif qty < min_qty:
            reason = "SIZE_TOO_SMALL"
        elif notional < min_notional:
            reason = "SIZE_TOO_SMALL"
        out.append(
            {
                "timestamp": str(row.get("timestamp", "")),
                "context": str(row.get("context", "")),
                "final_signal": int(sig),
                "qty_proxy": float(qty),
                "notional_proxy": float(notional),
                "live_reason": str(reason),
            }
        )
    return out


def _months(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    span = float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0)
    return max(span / 30.0, 1e-9)


def _seed_for(*parts: Any) -> int:
    return int(stable_hash("|".join(str(p) for p in parts), length=8), 16) % (2**31 - 1)
