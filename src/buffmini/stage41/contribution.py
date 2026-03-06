"""Stage-41 derivatives family contribution metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_family_contribution_metrics(
    *,
    layer_a: pd.DataFrame,
    layer_c: pd.DataFrame,
    stage_a_survivors: pd.DataFrame,
    stage_b_survivors: pd.DataFrame,
    families: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute per-family lift metrics through the Stage-39/40 funnel."""

    fams = [str(v) for v in (families or []) if str(v).strip()]
    if not fams:
        fams = sorted(
            {
                *set(layer_a.get("family", pd.Series(dtype=str)).astype(str).tolist()),
                *set(layer_c.get("family", pd.Series(dtype=str)).astype(str).tolist()),
                *set(stage_a_survivors.get("family", pd.Series(dtype=str)).astype(str).tolist()),
                *set(stage_b_survivors.get("family", pd.Series(dtype=str)).astype(str).tolist()),
            }
        )

    total_a = max(1, int(layer_a.shape[0]))
    total_c = max(1, int(layer_c.shape[0]))
    total_stage_a = max(1, int(stage_a_survivors.shape[0]))
    total_stage_b = max(1, int(stage_b_survivors.shape[0]))

    out: list[dict[str, Any]] = []
    for family in fams:
        a_count = int((layer_a.get("family", "").astype(str) == family).sum()) if not layer_a.empty else 0
        c_count = int((layer_c.get("family", "").astype(str) == family).sum()) if not layer_c.empty else 0
        sa_count = int((stage_a_survivors.get("family", "").astype(str) == family).sum()) if not stage_a_survivors.empty else 0
        sb_count = int((stage_b_survivors.get("family", "").astype(str) == family).sum()) if not stage_b_survivors.empty else 0

        candidate_lift = float(a_count / total_a)
        activation_lift = float(sa_count / max(1, c_count))
        tradability_lift = float(sb_count / max(1, sa_count))
        shortlist_share = float(c_count / total_c)
        final_policy_share = float(sb_count / total_stage_b)
        out.append(
            {
                "family": family,
                "layer_a_count": a_count,
                "layer_c_count": c_count,
                "stage_a_count": sa_count,
                "stage_b_count": sb_count,
                "candidate_lift": candidate_lift,
                "activation_lift": activation_lift,
                "tradability_lift": tradability_lift,
                "shortlist_share": shortlist_share,
                "final_policy_share": final_policy_share,
            }
        )
    return sorted(out, key=lambda row: (-int(row["stage_b_count"]), -int(row["layer_a_count"]), str(row["family"])))


def oi_short_only_runtime_guard(*, timeframe: str, short_only_enabled: bool, short_horizon_max: str) -> dict[str, Any]:
    """Return OI runtime activation policy for a timeframe."""

    allowed = _is_timeframe_shorter_or_equal(timeframe=timeframe, threshold=short_horizon_max)
    active = bool((not short_only_enabled) or allowed)
    return {
        "timeframe": str(timeframe),
        "short_only_enabled": bool(short_only_enabled),
        "short_horizon_max": str(short_horizon_max),
        "timeframe_allowed": bool(allowed),
        "oi_allowed": bool(active),
    }


def _is_timeframe_shorter_or_equal(*, timeframe: str, threshold: str) -> bool:
    minutes_map = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "1d": 1440,
    }
    tf = minutes_map.get(str(timeframe).strip().lower())
    cutoff = minutes_map.get(str(threshold).strip().lower())
    if tf is None or cutoff is None:
        return False
    return bool(tf <= cutoff)

