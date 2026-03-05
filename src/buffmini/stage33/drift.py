"""Drift monitoring utilities for Stage-33."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def representation_drift(
    baseline_embeddings: np.ndarray,
    recent_embeddings: np.ndarray,
    *,
    bins: int = 40,
) -> float:
    base = np.asarray(baseline_embeddings, dtype=float)
    recent = np.asarray(recent_embeddings, dtype=float)
    if base.size == 0 or recent.size == 0:
        return 0.0
    base_1d = base.reshape(-1)
    recent_1d = recent.reshape(-1)
    lo = float(min(np.min(base_1d), np.min(recent_1d)))
    hi = float(max(np.max(base_1d), np.max(recent_1d)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return 0.0
    hist_base, edges = np.histogram(base_1d, bins=int(max(10, bins)), range=(lo, hi), density=True)
    hist_recent, _ = np.histogram(recent_1d, bins=edges, density=True)
    p = hist_base + 1e-12
    q = hist_recent + 1e-12
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def performance_drift(
    *,
    baseline_metrics: dict[str, Any],
    recent_metrics: dict[str, Any],
) -> dict[str, float]:
    def _f(dic: dict[str, Any], key: str) -> float:
        return float(pd.to_numeric(pd.Series([dic.get(key, 0.0)]), errors="coerce").fillna(0.0).iloc[0])

    exp_lcb_delta = _f(recent_metrics, "exp_lcb") - _f(baseline_metrics, "exp_lcb")
    pf_delta = _f(recent_metrics, "PF_clipped") - _f(baseline_metrics, "PF_clipped")
    dd_delta = _f(recent_metrics, "maxDD") - _f(baseline_metrics, "maxDD")
    return {
        "exp_lcb_delta": float(exp_lcb_delta),
        "pf_delta": float(pf_delta),
        "maxdd_delta": float(dd_delta),
    }


def build_drift_summary(
    *,
    rep_drift: float,
    perf_drift: dict[str, float],
    rep_warn_threshold: float = 0.10,
    exp_lcb_warn_delta: float = -0.01,
) -> dict[str, Any]:
    warnings: list[str] = []
    if float(rep_drift) > float(rep_warn_threshold):
        warnings.append("representation_drift_high")
    if float(perf_drift.get("exp_lcb_delta", 0.0)) < float(exp_lcb_warn_delta):
        warnings.append("performance_drift_negative_exp_lcb")
    return {
        "representation_drift": float(rep_drift),
        "performance_drift": {
            "exp_lcb_delta": float(perf_drift.get("exp_lcb_delta", 0.0)),
            "pf_delta": float(perf_drift.get("pf_delta", 0.0)),
            "maxdd_delta": float(perf_drift.get("maxdd_delta", 0.0)),
        },
        "warnings": warnings,
    }

