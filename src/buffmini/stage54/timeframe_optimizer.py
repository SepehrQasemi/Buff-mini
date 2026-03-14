"""Stage-54 multi-timeframe discovery and deterministic optimizer helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.utils.hashing import stable_hash


def build_timeframe_metrics(
    candidates: pd.DataFrame,
    *,
    runtime_by_timeframe: dict[str, float] | None = None,
) -> pd.DataFrame:
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "candidate_count",
                "tradable_rate",
                "expected_net_after_cost_mean",
                "rr_adequacy_rate",
                "dominant_rejection_reason",
                "runtime_per_candidate",
                "promotion_score",
            ]
        )
    frame["timeframe"] = frame.get("timeframe", "").astype(str)
    frame["expected_net_after_cost"] = pd.to_numeric(frame.get("expected_net_after_cost", 0.0), errors="coerce").fillna(0.0)
    rr = _extract_rr(frame)
    frame["rr_ok"] = (rr >= 1.5).astype(int)
    if "stage_a_pass" not in frame.columns:
        stage_a = (
            (pd.to_numeric(frame.get("tp_before_sl_prob", 0.0), errors="coerce").fillna(0.0) >= 0.55)
            & (frame["expected_net_after_cost"] > 0.0)
            & (rr >= 1.5)
            & (pd.to_numeric(frame.get("cost_edge_proxy", 0.0), errors="coerce").fillna(0.0) > 0.0)
        )
        frame["stage_a_pass"] = stage_a.astype(int)
    runtimes = {str(key): float(value) for key, value in dict(runtime_by_timeframe or {}).items()}
    rows: list[dict[str, Any]] = []
    for timeframe, grp in frame.groupby("timeframe", dropna=False):
        rejection_series = grp.get("pre_replay_reject_reason", pd.Series(dtype=str)).astype(str)
        rejection_counts = rejection_series.loc[rejection_series != ""].value_counts(dropna=False)
        dominant_reason = str(rejection_counts.index[0]) if not rejection_counts.empty else ""
        runtime_per_candidate = float(runtimes.get(str(timeframe), 0.0) / max(1, len(grp)))
        tradable_rate = float(pd.to_numeric(grp.get("stage_a_pass", 0), errors="coerce").fillna(0).astype(int).mean())
        net_mean = float(grp["expected_net_after_cost"].mean())
        rr_rate = float(pd.to_numeric(grp["rr_ok"], errors="coerce").fillna(0).astype(int).mean())
        failure_penalty = 0.10 if dominant_reason else 0.0
        promotion_score = float((max(0.0, net_mean) * 100.0) + (tradable_rate * 0.40) + (rr_rate * 0.20) - (runtime_per_candidate * 0.02) - failure_penalty)
        rows.append(
            {
                "timeframe": str(timeframe),
                "candidate_count": int(len(grp)),
                "tradable_rate": float(round(tradable_rate, 8)),
                "expected_net_after_cost_mean": float(round(net_mean, 8)),
                "rr_adequacy_rate": float(round(rr_rate, 8)),
                "dominant_rejection_reason": dominant_reason,
                "runtime_per_candidate": float(round(runtime_per_candidate, 8)),
                "promotion_score": float(round(promotion_score, 8)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["promotion_score", "timeframe"], ascending=[False, True]).reset_index(drop=True)


def select_timeframe_promotions(
    metrics: pd.DataFrame,
    *,
    promotion_timeframes: int = 2,
    final_validation_timeframes: int = 1,
) -> dict[str, Any]:
    frame = metrics.copy() if isinstance(metrics, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return {
            "promotion_timeframes": [],
            "final_validation_timeframes": [],
            "summary_hash": stable_hash({"promotion_timeframes": [], "final_validation_timeframes": []}, length=16),
        }
    ranked = frame.sort_values(["promotion_score", "timeframe"], ascending=[False, True]).reset_index(drop=True)
    top_promote = ranked.head(int(max(1, promotion_timeframes)))["timeframe"].astype(str).tolist()
    top_final = ranked.head(int(max(1, final_validation_timeframes)))["timeframe"].astype(str).tolist()
    return {
        "promotion_timeframes": top_promote,
        "final_validation_timeframes": top_final,
        "summary_hash": stable_hash(
            {"promotion_timeframes": top_promote, "final_validation_timeframes": top_final},
            length=16,
        ),
    }


def hyperband_prune(
    candidates: pd.DataFrame,
    *,
    score_col: str = "replay_priority",
    keep: int = 20,
) -> pd.DataFrame:
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return frame
    frame[score_col] = pd.to_numeric(frame.get(score_col, 0.0), errors="coerce").fillna(0.0)
    frame = frame.sort_values([score_col, "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    frame["hyperband_keep"] = False
    frame.loc[frame.index[: int(max(1, keep))], "hyperband_keep"] = True
    frame["early_stopped"] = (~frame["hyperband_keep"]).astype(bool)
    return frame


def tpe_suggest(
    history: pd.DataFrame,
    *,
    search_space: dict[str, list[Any]],
    objective_col: str = "objective",
) -> dict[str, Any]:
    if not isinstance(history, pd.DataFrame) or history.empty:
        return {key: values[0] for key, values in sorted(search_space.items(), key=lambda kv: str(kv[0])) if values}
    work = history.copy()
    work[objective_col] = pd.to_numeric(work.get(objective_col, 0.0), errors="coerce").fillna(0.0)
    cutoff = max(1, int(np.ceil(len(work) * 0.25)))
    elite = work.sort_values([objective_col], ascending=False).head(cutoff)
    suggestion: dict[str, Any] = {}
    for name, values in sorted(search_space.items(), key=lambda kv: str(kv[0])):
        if not values:
            continue
        if name not in elite.columns:
            suggestion[name] = values[0]
            continue
        if isinstance(values[0], (int, float)):
            suggestion[name] = float(pd.to_numeric(elite[name], errors="coerce").dropna().median()) if not elite[name].dropna().empty else values[0]
        else:
            mode = elite[name].astype(str).mode(dropna=True)
            suggestion[name] = mode.iloc[0] if not mode.empty else values[0]
    return suggestion


def _extract_rr(frame: pd.DataFrame) -> pd.Series:
    if "first_target_rr" in frame.columns:
        return pd.to_numeric(frame["first_target_rr"], errors="coerce").fillna(0.0).astype(float)
    if "rr_model" not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    values: list[float] = []
    for raw in frame["rr_model"].tolist():
        if isinstance(raw, dict):
            values.append(float(raw.get("first_target_rr", 0.0)))
        else:
            values.append(0.0)
    return pd.Series(values, index=frame.index, dtype=float)
