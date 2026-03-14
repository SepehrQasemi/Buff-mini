"""Stage-68 uncertainty-aware gating."""

from __future__ import annotations

from typing import Any

import pandas as pd


def apply_uncertainty_gate_v3(
    candidates: pd.DataFrame,
    *,
    prob_col: str = "tp_before_sl_prob",
    net_col: str = "expected_net_after_cost",
    uncertainty_col: str = "uncertainty_score",
    max_uncertainty: float = 0.25,
    min_tp_before_sl_prob: float = 0.55,
    min_expected_net_after_cost: float = 0.0,
    allow_proxy_uncertainty: bool = False,
) -> dict[str, Any]:
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return {"gated": pd.DataFrame(), "counts": {"input": 0, "accepted": 0, "abstained": 0}}
    prob_src = frame[prob_col] if prob_col in frame.columns else pd.Series([0.0] * len(frame), index=frame.index)
    net_src = frame[net_col] if net_col in frame.columns else pd.Series([0.0] * len(frame), index=frame.index)
    frame[prob_col] = pd.to_numeric(prob_src, errors="coerce").fillna(0.0)
    frame[net_col] = pd.to_numeric(net_src, errors="coerce").fillna(0.0)
    uncertainty_source_type = "real"
    if uncertainty_col not in frame.columns:
        if bool(allow_proxy_uncertainty):
            # Deterministic fallback proxy from decision-boundary distance.
            frame[uncertainty_col] = (frame[prob_col] - 0.5).abs().rsub(0.5).clip(lower=0.0, upper=0.5)
            uncertainty_source_type = "proxy_only"
        else:
            frame[uncertainty_col] = 1.0
            uncertainty_source_type = "synthetic"
    frame[uncertainty_col] = pd.to_numeric(frame[uncertainty_col], errors="coerce").fillna(1.0)
    accepted_mask = (
        (frame[prob_col] >= float(min_tp_before_sl_prob))
        & (frame[net_col] > float(min_expected_net_after_cost))
        & (frame[uncertainty_col] <= float(max_uncertainty))
    )
    out = frame.loc[accepted_mask, :].copy().sort_values([prob_col, net_col], ascending=[False, False]).reset_index(drop=True)
    return {
        "gated": out,
        "counts": {
            "input": int(len(frame)),
            "accepted": int(len(out)),
            "abstained": int(len(frame) - len(out)),
        },
        "uncertainty_source_type": uncertainty_source_type,
        "allow_proxy_uncertainty": bool(allow_proxy_uncertainty),
    }
