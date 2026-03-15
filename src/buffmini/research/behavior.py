"""Behavior-aware candidate fingerprints for ranking and diagnostics."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from buffmini.utils.hashing import stable_hash
from buffmini.validation.candidate_runtime import build_candidate_signal_series, hydrate_candidate_record


def build_behavioral_fingerprints(
    candidates: pd.DataFrame,
    market_frame: pd.DataFrame,
    *,
    max_candidates: int = 96,
    max_bars: int = 4096,
) -> pd.DataFrame:
    """Profile candidate behavior from actual executable signal series on one market frame."""

    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if frame.empty or market_frame is None or market_frame.empty:
        return pd.DataFrame(columns=["candidate_id"])
    frame = _select_behavior_candidates(frame, max_candidates=max_candidates)

    bars = market_frame.copy()
    bars["timestamp"] = pd.to_datetime(bars.get("timestamp"), utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        bars[col] = pd.to_numeric(bars.get(col), errors="coerce").astype(float)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    if int(max_bars) > 0 and len(bars) > int(max_bars):
        bars = bars.tail(int(max_bars)).reset_index(drop=True)
    next_return = pd.to_numeric(bars["close"], errors="coerce").pct_change().shift(-1).fillna(0.0)
    regime_labels = _derive_regime_labels(bars)

    exit_keys: list[str] = []
    signal_store: dict[str, pd.Series] = {}
    active_store: dict[str, pd.Series] = {}
    pnl_store: dict[str, pd.Series] = {}
    base_rows: list[dict[str, Any]] = []

    for row in frame.to_dict(orient="records"):
        candidate = hydrate_candidate_record(dict(row))
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if not candidate_id:
            continue
        signal = build_candidate_signal_series(bars, candidate).reindex(bars.index).fillna(0).astype(int)
        active = signal.ne(0)
        active_count = int(active.sum())
        long_count = int((signal > 0).sum())
        short_count = int((signal < 0).sum())
        active_denom = max(1, active_count)
        activation_density = float(active.mean())
        long_share = float(long_count / active_denom)
        short_share = float(short_count / active_denom)
        pseudo_pnl = signal.astype(float) * next_return.astype(float)
        regime_map = _regime_activation_map(active=active, regime_labels=regime_labels)
        exit_key = "|".join(
            [
                str(candidate.get("exit_family", candidate.get("target_logic", ""))),
                str(candidate.get("risk_model", "")),
                str(candidate.get("time_stop_bars", candidate.get("expected_hold_bars", 0))),
            ]
        )
        exit_keys.append(exit_key)
        signal_store[candidate_id] = signal
        active_store[candidate_id] = active
        pnl_store[candidate_id] = pseudo_pnl
        base_rows.append(
            {
                "candidate_id": candidate_id,
                "activation_density": float(round(activation_density, 8)),
                "active_count": int(active_count),
                "side_distribution": json.dumps(
                    {"long_share": round(long_share, 6), "short_share": round(short_share, 6)},
                    sort_keys=True,
                ),
                "hold_distribution": json.dumps(
                    {"expected_hold_bars": int(candidate.get("time_stop_bars", candidate.get("expected_hold_bars", 0)) or 0)},
                    sort_keys=True,
                ),
                "regime_activation_map": json.dumps(regime_map, sort_keys=True),
                "exit_key": exit_key,
            }
        )

    out = pd.DataFrame(base_rows)
    if out.empty:
        return out

    exit_key_counts = out["exit_key"].astype(str).value_counts().to_dict()
    ids = out["candidate_id"].astype(str).tolist()
    for idx, candidate_id in enumerate(ids):
        active_i = active_store[candidate_id]
        signal_i = signal_store[candidate_id]
        pnl_i = pnl_store[candidate_id]
        entry_overlaps: list[float] = []
        pnl_corrs: list[float] = []
        signal_corrs: list[float] = []
        for other_id in ids:
            if other_id == candidate_id:
                continue
            active_j = active_store[other_id]
            signal_j = signal_store[other_id]
            pnl_j = pnl_store[other_id]
            entry_overlaps.append(_jaccard(active_i, active_j))
            signal_corrs.append(_safe_corr(signal_i.astype(float), signal_j.astype(float)))
            pnl_corrs.append(_safe_corr(pnl_i.astype(float), pnl_j.astype(float)))
        exit_share = float((int(exit_key_counts.get(str(out.iloc[idx]["exit_key"]), 1)) - 1) / max(1, len(ids) - 1))
        max_entry_overlap = max(entry_overlaps) if entry_overlaps else 0.0
        max_signal_corr = max(signal_corrs) if signal_corrs else 0.0
        max_pnl_corr = max(pnl_corrs) if pnl_corrs else 0.0
        clustering_risk = float(np.mean(sorted([max_entry_overlap, max_signal_corr, max_pnl_corr, exit_share], reverse=True)[:3]))
        failure_pattern_similarity = float((max_entry_overlap + exit_share + max_pnl_corr) / 3.0)
        behavioral_fingerprint = stable_hash(
            {
                "candidate_id": candidate_id,
                "activation_density": round(float(out.iloc[idx]["activation_density"]), 6),
                "side_distribution": json.loads(str(out.iloc[idx]["side_distribution"])),
                "regime_activation_map": json.loads(str(out.iloc[idx]["regime_activation_map"])),
                "entry_overlap_score": round(max_entry_overlap, 6),
                "exit_overlap_score": round(exit_share, 6),
                "pnl_correlation_risk": round(max_pnl_corr, 6),
            },
            length=20,
        )
        out.loc[idx, "entry_overlap_score"] = float(round(max_entry_overlap, 8))
        out.loc[idx, "exit_overlap_score"] = float(round(exit_share, 8))
        out.loc[idx, "pnl_correlation_risk"] = float(round(max_pnl_corr, 8))
        out.loc[idx, "clustering_risk"] = float(round(clustering_risk, 8))
        out.loc[idx, "failure_pattern_similarity"] = float(round(failure_pattern_similarity, 8))
        out.loc[idx, "behavioral_fingerprint"] = str(behavioral_fingerprint)
    return out.drop(columns=["exit_key"]).reset_index(drop=True)


def _select_behavior_candidates(frame: pd.DataFrame, *, max_candidates: int) -> pd.DataFrame:
    if frame.empty or len(frame) <= int(max_candidates):
        return frame.copy()

    work = frame.copy()
    layer = pd.to_numeric(work.get("beam_score", work.get("layer_score", 0.0)), errors="coerce").fillna(0.0)
    exp_lcb = pd.to_numeric(work.get("exp_lcb_proxy", 0.0), errors="coerce").fillna(0.0)
    cost_edge = pd.to_numeric(work.get("cost_edge_proxy", 0.0), errors="coerce").fillna(0.0)
    rr_values = []
    for raw_rr in work.get("rr_model", pd.Series([{}] * len(work), index=work.index)).tolist():
        rr_model = raw_rr if isinstance(raw_rr, dict) else {}
        rr_values.append(float(rr_model.get("first_target_rr", 0.0)))
    rr = pd.to_numeric(pd.Series(rr_values, index=work.index), errors="coerce").fillna(0.0)
    work["_behavior_priority"] = (layer * 0.45) + (exp_lcb * 40.0 * 0.25) + (cost_edge * 100.0 * 0.20) + ((rr.clip(lower=0.0, upper=3.0) / 3.0) * 0.10)
    work["_family"] = work.get("family", "").astype(str)
    family_count = max(1, int(work["_family"].nunique()))
    per_family_quota = max(2, int(max_candidates) // family_count)
    chosen_idx: list[int] = []
    for family, group in work.groupby("_family", sort=True):
        del family
        ordered = group.sort_values(["_behavior_priority", "candidate_id"], ascending=[False, True]).head(per_family_quota)
        chosen_idx.extend(int(idx) for idx in ordered.index.tolist())
    remaining = max(0, int(max_candidates) - len(set(chosen_idx)))
    if remaining > 0:
        already = set(chosen_idx)
        extras = (
            work.loc[~work.index.isin(already)]
            .sort_values(["_behavior_priority", "candidate_id"], ascending=[False, True])
            .head(remaining)
        )
        chosen_idx.extend(int(idx) for idx in extras.index.tolist())
    out = work.loc[sorted(set(chosen_idx))].copy()
    return out.drop(columns=["_behavior_priority", "_family"], errors="ignore").reset_index(drop=True)


def _derive_regime_labels(bars: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(bars.get("close"), errors="coerce").astype(float)
    returns = close.pct_change().fillna(0.0)
    drift = close / close.rolling(24, min_periods=4).mean().replace(0.0, np.nan) - 1.0
    vol = returns.rolling(12, min_periods=4).std().fillna(0.0)
    compression_cutoff = float(vol.quantile(0.30)) if len(vol) else 0.0
    shock_cutoff = float(vol.quantile(0.80)) if len(vol) else 0.0
    labels = pd.Series("range", index=bars.index, dtype=object)
    labels.loc[drift > 0.01] = "trend_up"
    labels.loc[drift < -0.01] = "trend_down"
    labels.loc[vol <= compression_cutoff] = "compression"
    labels.loc[vol >= shock_cutoff] = "shock"
    return labels.astype(str)


def _regime_activation_map(*, active: pd.Series, regime_labels: pd.Series) -> dict[str, float]:
    active_mask = active.fillna(False).astype(bool)
    if not bool(active_mask.any()):
        return {label: 0.0 for label in sorted(regime_labels.astype(str).unique().tolist())}
    counts = regime_labels.loc[active_mask].astype(str).value_counts(normalize=True).to_dict()
    labels = sorted(regime_labels.astype(str).unique().tolist())
    return {label: float(round(counts.get(label, 0.0), 6)) for label in labels}


def _jaccard(left: pd.Series, right: pd.Series) -> float:
    left_mask = left.fillna(False).astype(bool)
    right_mask = right.fillna(False).astype(bool)
    union = int((left_mask | right_mask).sum())
    if union <= 0:
        return 0.0
    intersection = int((left_mask & right_mask).sum())
    return float(intersection / union)


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    a = pd.to_numeric(left, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    b = pd.to_numeric(right, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if len(a) < 3 or float(a.std(ddof=0)) <= 1e-12 or float(b.std(ddof=0)) <= 1e-12:
        return 0.0
    corr = float(a.corr(b))
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))
