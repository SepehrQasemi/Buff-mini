"""Stage-16 context state engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


STATES = ["TREND", "RANGE", "VOL_EXPANSION", "VOL_COMPRESSION", "CHOP"]


@dataclass(frozen=True)
class ContextConfig:
    trend_threshold: float = 0.010
    vol_high: float = 0.75
    vol_low: float = 0.25
    chop_flip_window: int = 48
    chop_flip_threshold: float = 0.20


def compute_context_states(frame: pd.DataFrame, cfg: ContextConfig | None = None) -> pd.DataFrame:
    """Compute deterministic context label + score columns."""

    c = cfg or ContextConfig()
    out = frame.copy()
    close = pd.to_numeric(out.get("close", 0.0), errors="coerce").fillna(0.0)
    atr_rank = pd.to_numeric(out.get("atr_pct_rank_252", 0.5), errors="coerce").fillna(0.5)
    bb_z = pd.to_numeric(out.get("bb_bandwidth_z_120", 0.0), errors="coerce").fillna(0.0)
    slope = pd.to_numeric(out.get("ema_slope_50", 0.0), errors="coerce").fillna(0.0)
    trend_strength = pd.to_numeric(out.get("trend_strength_stage10", slope.abs()), errors="coerce").fillna(slope.abs())

    trend_score = np.clip((trend_strength / max(c.trend_threshold, 1e-9)).to_numpy(dtype=float), 0.0, 1.0)
    range_score = np.clip((1.0 - trend_score) * (1.0 - np.abs(atr_rank.to_numpy(dtype=float) - 0.5) * 2.0), 0.0, 1.0)
    vol_exp_score = np.clip(np.maximum(atr_rank.to_numpy(dtype=float) - c.vol_high, 0.0) / max(1e-9, 1.0 - c.vol_high), 0.0, 1.0)
    vol_cmp_score = np.clip(np.maximum(c.vol_low - atr_rank.to_numpy(dtype=float), 0.0) / max(1e-9, c.vol_low), 0.0, 1.0)

    direction = np.sign(close.diff().fillna(0.0).to_numpy(dtype=float))
    flips = (direction != np.roll(direction, 1)).astype(float)
    flips[0] = 0.0
    chop_rate = pd.Series(flips, index=out.index).rolling(c.chop_flip_window, min_periods=4).mean().fillna(0.0)
    chop_score = np.clip((chop_rate.to_numpy(dtype=float) - c.chop_flip_threshold) / max(1e-9, 1.0 - c.chop_flip_threshold), 0.0, 1.0)
    chop_score = chop_score * np.clip(1.0 - np.abs(atr_rank.to_numpy(dtype=float) - 0.5) * 2.0, 0.0, 1.0)

    scores = np.column_stack([trend_score, range_score, vol_exp_score, vol_cmp_score, chop_score])
    label_idx = np.argmax(scores, axis=1)
    labels = np.asarray(STATES, dtype=object)[label_idx]
    confidence = np.max(scores, axis=1)

    out["ctx_score_trend"] = trend_score
    out["ctx_score_range"] = range_score
    out["ctx_score_vol_expansion"] = vol_exp_score
    out["ctx_score_vol_compression"] = vol_cmp_score
    out["ctx_score_chop"] = chop_score
    out["ctx_state"] = pd.Series(labels, index=out.index, dtype="object")
    out["ctx_confidence"] = pd.Series(confidence, index=out.index, dtype=float)
    return out


def context_persistence_summary(frame_with_context: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return duration summary and transition matrix."""

    state = frame_with_context.get("ctx_state")
    if state is None or len(state) == 0:
        return pd.DataFrame(columns=["state", "p50_duration", "p90_duration"]), pd.DataFrame(
            np.zeros((len(STATES), len(STATES)), dtype=float), index=STATES, columns=STATES
        )
    s = state.astype(str).reset_index(drop=True)
    grp = (s != s.shift(1)).cumsum()
    run_len = grp.groupby(grp).transform("size").astype(int)
    run_state = s.groupby(grp).first().astype(str)

    duration_rows = []
    for st in STATES:
        d = run_len.loc[run_state[grp].to_numpy(dtype=bool) if False else (run_state.reindex(grp).to_numpy(dtype=str) == st)]
        # simpler deterministic extraction:
        vals = run_len[s == st].to_numpy(dtype=float)
        if vals.size == 0:
            p50 = 0.0
            p90 = 0.0
        else:
            p50 = float(np.percentile(vals, 50))
            p90 = float(np.percentile(vals, 90))
        duration_rows.append({"state": st, "p50_duration": p50, "p90_duration": p90})

    trans = pd.DataFrame(np.zeros((len(STATES), len(STATES)), dtype=float), index=STATES, columns=STATES)
    prev = s.shift(1)
    for a, b in zip(prev.iloc[1:], s.iloc[1:]):
        if a in STATES and b in STATES:
            trans.loc[str(a), str(b)] += 1.0
    row_sum = trans.sum(axis=1).replace(0.0, 1.0)
    trans = trans.div(row_sum, axis=0)
    return pd.DataFrame(duration_rows), trans


def context_weight_series(frame_with_context: pd.DataFrame) -> pd.Series:
    """Convert context state/confidence to soft weight series."""

    state = frame_with_context.get("ctx_state", pd.Series("RANGE", index=frame_with_context.index)).astype(str)
    conf = pd.to_numeric(frame_with_context.get("ctx_confidence", 0.5), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    base = np.where(state == "TREND", 1.05, np.where(state == "RANGE", 1.00, np.where(state == "CHOP", 0.90, 0.95)))
    return pd.Series(np.clip(1.0 + (base - 1.0) * conf.to_numpy(dtype=float), 0.80, 1.20), index=frame_with_context.index, dtype=float)

