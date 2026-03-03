"""Stage-27.6 contextual edge interpretation logic."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def contextual_edge_verdict(
    rolling_results: pd.DataFrame,
    *,
    min_windows: int = 3,
    robust_repeat_rate: float = 0.60,
) -> dict[str, Any]:
    """Classify contextual edge stability from rolling discovery rows."""

    if rolling_results.empty:
        return {
            "rows": [],
            "policy_verdict": "DOES_POLICY_HAVE_CONTEXTUAL_EDGE",
            "has_contextual_edge": False,
        }

    work = rolling_results.copy()
    context_col = "best_context" if "best_context" in work.columns else "context"
    rulelet_col = "best_rulelet" if "best_rulelet" in work.columns else "rulelet"
    tf_col = "timeframe"
    exp_col = "best_exp_lcb" if "best_exp_lcb" in work.columns else "exp_lcb"
    trades_col = "best_trades_in_context" if "best_trades_in_context" in work.columns else "trades_in_context"

    work[context_col] = work.get(context_col, "").astype(str)
    work[rulelet_col] = work.get(rulelet_col, "").astype(str)
    work[tf_col] = work.get(tf_col, "").astype(str)
    work[exp_col] = pd.to_numeric(work.get(exp_col, 0.0), errors="coerce").fillna(0.0)
    work[trades_col] = pd.to_numeric(work.get(trades_col, 0.0), errors="coerce").fillna(0.0)

    rows: list[dict[str, Any]] = []
    for keys, grp in work.groupby([context_col, rulelet_col, tf_col], dropna=False):
        context, rulelet, timeframe = keys
        total = int(grp.shape[0])
        positive = int((grp[exp_col] > 0.0).sum())
        repeat_rate = float(positive / max(1, total))
        median_exp_lcb = float(grp[exp_col].median()) if total else 0.0
        median_trades = float(grp[trades_col].median()) if total else 0.0
        if total >= int(min_windows) and repeat_rate >= float(robust_repeat_rate) and median_exp_lcb > 0.0:
            verdict = "ROBUST_IN_CONTEXT"
        elif positive > 0 and median_exp_lcb > 0.0:
            verdict = "WEAK"
        else:
            verdict = "NOISE"
        rows.append(
            {
                "context": str(context),
                "rulelet": str(rulelet),
                "timeframe": str(timeframe),
                "windows_total": int(total),
                "windows_positive": int(positive),
                "repeat_rate": float(repeat_rate),
                "median_exp_lcb": float(median_exp_lcb),
                "median_trades": float(median_trades),
                "verdict": str(verdict),
            }
        )

    verdict_df = pd.DataFrame(rows)
    robust_count = int((verdict_df["verdict"] == "ROBUST_IN_CONTEXT").sum()) if not verdict_df.empty else 0
    weak_count = int((verdict_df["verdict"] == "WEAK").sum()) if not verdict_df.empty else 0
    has_edge = bool(robust_count > 0 or weak_count >= 2)
    return {
        "rows": rows,
        "policy_verdict": "DOES_POLICY_HAVE_CONTEXTUAL_EDGE" if has_edge else "NO_CONTEXTUAL_EDGE",
        "has_contextual_edge": has_edge,
        "robust_count": int(robust_count),
        "weak_count": int(weak_count),
        "noise_count": int((verdict_df["verdict"] == "NOISE").sum()) if not verdict_df.empty else 0,
    }

