"""Stage-26 conditional policy composer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_conditional_policy(
    *,
    effects: pd.DataFrame,
    min_occurrences_per_context: int = 30,
    min_trades_in_context: int = 30,
    top_k: int = 2,
    w_min: float = 0.05,
    w_max: float = 0.80,
) -> dict[str, Any]:
    """Build context->rulelet weighted policy from conditional effects."""

    if effects.empty:
        return {"contexts": {}, "warnings": ["no_effect_rows"]}
    work = effects.copy()
    work["context_occurrences"] = pd.to_numeric(work.get("context_occurrences", 0.0), errors="coerce").fillna(0.0)
    work["trades_in_context"] = pd.to_numeric(work.get("trades_in_context", 0.0), errors="coerce").fillna(0.0)
    work["exp_lcb"] = pd.to_numeric(work.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
    work["classification"] = work.get("classification", "FAIL").astype(str)

    contexts: dict[str, Any] = {}
    warnings: list[str] = []
    for context, grp in work.groupby("context", dropna=False):
        context_name = str(context)
        g = grp.loc[grp["context_occurrences"] >= float(min_occurrences_per_context)].copy()
        if g.empty:
            warnings.append(f"{context_name}:insufficient_occurrences")
            contexts[context_name] = {"rulelets": [], "weights": {}, "status": "EMPTY"}
            continue
        g = g.loc[
            (g["trades_in_context"] >= float(min_trades_in_context))
            | (g["classification"].astype(str) == "RARE")
        ].copy()
        if g.empty:
            warnings.append(f"{context_name}:insufficient_trades")
            contexts[context_name] = {"rulelets": [], "weights": {}, "status": "EMPTY"}
            continue
        ranked = g.sort_values(["exp_lcb", "expectancy"], ascending=[False, False]).head(int(max(1, top_k)))
        pos = np.maximum(pd.to_numeric(ranked["exp_lcb"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0)
        if float(pos.sum()) <= 0.0:
            # Fallback to equal weights across retained rulelets with non-positive LCB.
            pos = np.ones(ranked.shape[0], dtype=float)
        raw = pos / max(1e-12, float(pos.sum()))
        clipped = np.clip(raw, float(w_min), float(w_max))
        clipped = clipped / max(1e-12, float(clipped.sum()))
        chosen = ranked["rulelet"].astype(str).tolist()
        weights = {name: float(weight) for name, weight in zip(chosen, clipped, strict=False)}
        contexts[context_name] = {
            "rulelets": chosen,
            "weights": weights,
            "status": "OK",
            "source_rows": ranked[["rulelet", "family", "exp_lcb", "trades_in_context", "classification"]].to_dict(orient="records"),
        }
    return {"contexts": contexts, "warnings": warnings}


def compose_policy_signal(
    *,
    frame: pd.DataFrame,
    rulelet_scores: dict[str, pd.Series],
    policy: dict[str, Any],
    conflict_mode: str = "net",
) -> tuple[pd.Series, pd.DataFrame]:
    """Compose single policy signal from context-selected rulelet scores."""

    mode = str(conflict_mode).strip().lower()
    state = frame.get("ctx_state", pd.Series("RANGE", index=frame.index)).astype(str)
    rows: list[dict[str, Any]] = []
    signal = np.zeros(len(frame), dtype=int)
    contexts = dict(policy.get("contexts", {}))
    for i in range(len(frame)):
        ctx = str(state.iloc[i])
        ctx_policy = dict(contexts.get(ctx, {}))
        weights = dict(ctx_policy.get("weights", {}))
        long_score = 0.0
        short_score = 0.0
        net_score = 0.0
        for rulelet, weight in weights.items():
            series = pd.to_numeric(rulelet_scores.get(str(rulelet), pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
            val = float(series.iloc[i]) if i < len(series) else 0.0
            w = float(weight)
            net_score += w * val
            if val > 0:
                long_score += w * val
            elif val < 0:
                short_score += w * abs(val)
        if mode == "hedge":
            s = 1 if long_score >= short_score else -1
            if max(long_score, short_score) <= 0.0:
                s = 0
        elif mode == "isolated":
            s = 1 if net_score > 0 else -1 if net_score < 0 else 0
        else:
            s = 1 if net_score > 0 else -1 if net_score < 0 else 0
        signal[i] = int(s)
        rows.append(
            {
                "timestamp": str(pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").iloc[i]) if "timestamp" in frame.columns else "",
                "context": ctx,
                "long_score": float(long_score),
                "short_score": float(short_score),
                "net_score": float(net_score),
                "final_signal": int(s),
                "conflict_mode": mode,
            }
        )
    sig_series = pd.Series(signal, index=frame.index, dtype=int).shift(1).fillna(0).astype(int)
    trace = pd.DataFrame(rows)
    return sig_series, trace

