"""Stage-32 feasibility envelope integration helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.execution.feasibility import min_required_risk_pct


def candidate_feasibility_envelope(
    *,
    signals: pd.DataFrame,
    equity_tiers: list[float],
    min_notional: float,
    cost_rt_pct: float,
    max_notional_pct: float,
    risk_cap: float,
) -> pd.DataFrame:
    """Compute feasibility envelope per candidate and aggregate policy view."""

    if signals.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "symbol",
                "timeframe",
                "context",
                "equity",
                "signals",
                "feasible_pct",
                "min_required_risk_p50",
                "min_required_risk_p90",
                "recommended_risk_floor",
            ]
        )
    work = signals.copy()
    for col, default in (
        ("candidate_id", "policy"),
        ("symbol", ""),
        ("timeframe", ""),
        ("context", "UNKNOWN"),
    ):
        if col not in work.columns:
            work[col] = default
        work[col] = work[col].astype(str)
    work["stop_dist_pct"] = pd.to_numeric(work.get("stop_dist_pct", 0.0), errors="coerce").fillna(0.0)
    work = work.loc[work["stop_dist_pct"] > 0.0].copy()
    if work.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = work.groupby(["candidate_id", "symbol", "timeframe", "context"], dropna=False)
    for keys, grp in grouped:
        cand, sym, tf, ctx = keys
        stop_values = grp["stop_dist_pct"].to_numpy(dtype=float)
        for equity in [float(v) for v in equity_tiers]:
            req = np.asarray(
                [
                    min_required_risk_pct(
                        equity=float(equity),
                        min_notional=float(min_notional),
                        stop_dist_pct=float(stop),
                        cost_rt_pct=float(cost_rt_pct),
                        max_notional_pct=float(max_notional_pct),
                    )
                    for stop in stop_values
                ],
                dtype=float,
            )
            req = req[np.isfinite(req)]
            if req.size == 0:
                continue
            feasible = req <= float(risk_cap)
            rows.append(
                {
                    "candidate_id": str(cand),
                    "symbol": str(sym),
                    "timeframe": str(tf),
                    "context": str(ctx),
                    "equity": float(equity),
                    "signals": int(req.size),
                    "feasible_pct": float(np.mean(feasible) * 100.0),
                    "min_required_risk_p50": float(np.quantile(req, 0.50)),
                    "min_required_risk_p90": float(np.quantile(req, 0.90)),
                    "recommended_risk_floor": float(np.quantile(req, 0.75)),
                }
            )
    return pd.DataFrame(rows)

