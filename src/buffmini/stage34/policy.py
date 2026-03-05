"""Stage-34 policy selection and deterministic replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.execution.feasibility import min_required_risk_pct
from buffmini.stage34.train import predict_model_proba
from buffmini.utils.hashing import stable_hash


@dataclass(frozen=True)
class PolicyConfig:
    threshold: float = 0.55
    risk_cap: float = 0.20
    equity: float = 10_000.0
    live_min_notional: float = 10.0
    research_min_notional: float = 1.0
    max_notional_pct: float = 1.0


def select_best_policy(eval_rows: pd.DataFrame, *, cfg: PolicyConfig, seed: int) -> dict[str, Any]:
    work = eval_rows.copy() if isinstance(eval_rows, pd.DataFrame) else pd.DataFrame()
    if work.empty:
        return {
            "policy_id": f"stage34_{stable_hash({'empty': True, 'seed': int(seed)}, length=12)}",
            "status": "EMPTY",
            "selection_reason": "no_eval_rows",
            "threshold": float(cfg.threshold),
        }
    work["exp_lcb"] = pd.to_numeric(work.get("exp_lcb"), errors="coerce").fillna(0.0)
    work["trade_count"] = pd.to_numeric(work.get("trade_count"), errors="coerce").fillna(0).astype(int)
    work["positive_windows_ratio"] = pd.to_numeric(work.get("positive_windows_ratio"), errors="coerce").fillna(0.0)
    work["wf_executed"] = work.get("wf_executed", False).astype(bool)
    work["mc_triggered"] = work.get("mc_triggered", False).astype(bool)
    live = work.loc[work.get("cost_mode", "").astype(str) == "live"].copy()
    scope = live if not live.empty else work
    qualified = scope.loc[(scope["wf_executed"]) & (scope["mc_triggered"])].copy()
    if qualified.empty:
        qualified = scope.copy()
    best = qualified.sort_values(["exp_lcb", "positive_windows_ratio", "trade_count"], ascending=[False, False, False]).head(1)
    rec = best.iloc[0].to_dict()
    rare = int(rec.get("trade_count", 0)) < 30 and float(rec.get("exp_lcb", 0.0)) > 0.0
    status = "RARE_EDGE" if rare else ("EDGE" if float(rec.get("exp_lcb", 0.0)) > 0.0 else "NO_EDGE")
    payload = {
        "policy_id": f"stage34_{stable_hash({'seed': int(seed), 'model': rec.get('model_name', ''), 'threshold': float(cfg.threshold), 'exp_lcb': float(rec.get('exp_lcb', 0.0))}, length=12)}",
        "status": status,
        "selection_reason": "exp_lcb_then_stability",
        "model_name": str(rec.get("model_name", "")),
        "threshold": float(cfg.threshold),
        "source_row": {
            "window_months": int(rec.get("window_months", 0)),
            "cost_mode": str(rec.get("cost_mode", "")),
            "exp_lcb": float(rec.get("exp_lcb", 0.0)),
            "trade_count": int(rec.get("trade_count", 0)),
            "positive_windows_ratio": float(rec.get("positive_windows_ratio", 0.0)),
            "wf_executed": bool(rec.get("wf_executed", False)),
            "mc_triggered": bool(rec.get("mc_triggered", False)),
        },
    }
    return payload


def replay_policy(
    dataset: pd.DataFrame,
    *,
    model: dict[str, Any],
    policy: dict[str, Any],
    mode: str,
    cfg: PolicyConfig,
) -> dict[str, Any]:
    if dataset.empty:
        return {"status": "EMPTY_DATASET", "trade_count": 0}
    work = dataset.copy().sort_values("timestamp").reset_index(drop=True)
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    proba = predict_model_proba(model, work)
    threshold = float(policy.get("threshold", cfg.threshold))
    score = 2.0 * np.asarray(proba, dtype=float) - 1.0
    proposal = np.zeros(work.shape[0], dtype=int)
    proposal[proba >= threshold] = 1
    proposal[proba <= (1.0 - threshold)] = -1

    accepted = np.zeros_like(proposal, dtype=int)
    reject_reasons: dict[str, int] = {}
    shadow_live_reject = 0
    cost_rt = 0.001 if str(mode).lower() == "research" else 0.0015
    min_notional_live = float(cfg.live_min_notional)
    min_notional_research = float(cfg.research_min_notional)
    stop_dist = pd.to_numeric(work.get("atr_pct"), errors="coerce").fillna(0.0).clip(lower=1e-5).to_numpy(dtype=float)

    for i, sig in enumerate(proposal):
        if int(sig) == 0:
            continue
        req_live = min_required_risk_pct(
            equity=float(cfg.equity),
            min_notional=float(min_notional_live),
            stop_dist_pct=float(stop_dist[i]),
            cost_rt_pct=float(cost_rt),
            max_notional_pct=float(cfg.max_notional_pct),
        )
        if str(mode).lower() == "research":
            accepted[i] = int(sig)
            if np.isfinite(req_live) and req_live > float(cfg.risk_cap):
                shadow_live_reject += 1
            continue
        req = min_required_risk_pct(
            equity=float(cfg.equity),
            min_notional=float(min_notional_live if str(mode).lower() == "live" else min_notional_research),
            stop_dist_pct=float(stop_dist[i]),
            cost_rt_pct=float(cost_rt),
            max_notional_pct=float(cfg.max_notional_pct),
        )
        if np.isfinite(req) and req <= float(cfg.risk_cap):
            accepted[i] = int(sig)
        else:
            key = "SIZE_TOO_SMALL"
            reject_reasons[key] = int(reject_reasons.get(key, 0) + 1)

    signal = pd.Series(accepted, index=work.index, dtype=int).shift(1).fillna(0).astype(int)
    work["signal"] = signal
    result = run_backtest(
        frame=work,
        strategy_name=f"Stage34Policy::{policy.get('policy_id', '')}",
        symbol=str(work.get("symbol", pd.Series(["BTC/USDT"])).iloc[0] if "symbol" in work.columns else "BTC/USDT"),
        signal_col="signal",
        stop_atr_multiple=1.5,
        take_profit_atr_multiple=3.0,
        max_hold_bars=24,
        round_trip_cost_pct=0.05 if str(mode).lower() == "research" else 0.10,
        slippage_pct=0.0003 if str(mode).lower() == "research" else 0.0005,
        exit_mode="fixed_atr",
        cost_model_cfg={},
    )
    trades = result.trades.copy()
    proposals = int(np.count_nonzero(proposal))
    accepted_count = int(np.count_nonzero(accepted))
    rejected_count = int(max(0, proposals - accepted_count))
    top_rejects = sorted(reject_reasons.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "status": "OK",
        "mode": str(mode),
        "policy_id": str(policy.get("policy_id", "")),
        "proposal_count": proposals,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "accepted_rejected_breakdown": {
            "accepted": int(accepted_count),
            "rejected": int(rejected_count),
        },
        "top_reject_reasons": [{ "reason": str(k), "count": int(v)} for k, v in top_rejects[:5]],
        "shadow_live_reject_count": int(shadow_live_reject),
        "trade_count": int(trades.shape[0]),
        "expectancy": float(result.metrics.get("expectancy", 0.0)),
        "exp_lcb_proxy": float(np.percentile(pd.to_numeric(trades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy(dtype=float), 5))
        if not trades.empty
        else 0.0,
        "profit_factor": float(result.metrics.get("profit_factor", 0.0)),
        "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
    }
