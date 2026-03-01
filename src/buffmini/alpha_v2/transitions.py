"""Stage-19 transition signal library."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.alpha_v2.contracts import SignalContract


def transition_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Mechanism-based transition scores in [-1,1]."""

    out = pd.DataFrame(index=frame.index)
    close = pd.to_numeric(frame.get("close", 0.0), errors="coerce").fillna(0.0)
    atr_rank = pd.to_numeric(frame.get("atr_pct_rank_252", 0.5), errors="coerce").fillna(0.5)
    bw_z = pd.to_numeric(frame.get("bb_bandwidth_z_120", 0.0), errors="coerce").fillna(0.0)
    slope = pd.to_numeric(frame.get("ema_slope_50", 0.0), errors="coerce").fillna(0.0)

    compression = bw_z < -0.8
    expansion = bw_z > 0.8
    cmp_to_exp = compression.shift(1, fill_value=False) & expansion
    breakout_dir = np.sign(close.diff().fillna(0.0).to_numpy(dtype=float))
    out["tr_cmp_to_exp_breakout"] = SignalContract.clip_scores(
        pd.Series(cmp_to_exp.to_numpy(dtype=float) * breakout_dir, index=frame.index, dtype=float)
    )

    exhaustion = (slope.abs() > 0.01) & (atr_rank > 0.85)
    snapback = np.where(exhaustion & (slope > 0), -1.0, np.where(exhaustion & (slope < 0), 1.0, 0.0))
    out["tr_trend_exhaustion_snapback"] = SignalContract.clip_scores(
        pd.Series(snapback, index=frame.index, dtype=float)
    )

    vol_shock = atr_rank > 0.9
    delayed_drift = np.sign(close.diff(3).fillna(0.0).to_numpy(dtype=float))
    out["tr_vol_shock_delayed_drift"] = SignalContract.clip_scores(
        pd.Series(np.where(vol_shock, delayed_drift, 0.0), index=frame.index, dtype=float)
    )

    high_20 = pd.to_numeric(frame.get("donchian_high_20", close.rolling(20, min_periods=1).max()), errors="coerce")
    low_20 = pd.to_numeric(frame.get("donchian_low_20", close.rolling(20, min_periods=1).min()), errors="coerce")
    false_up = (close.shift(1) > high_20.shift(1)) & (close <= high_20)
    false_dn = (close.shift(1) < low_20.shift(1)) & (close >= low_20)
    out["tr_range_failure_reversal"] = SignalContract.clip_scores(
        pd.Series(false_dn.astype(float) - false_up.astype(float), index=frame.index, dtype=float)
    )
    return out


def combined_transition_score(frame: pd.DataFrame) -> pd.Series:
    table = transition_scores(frame)
    score = table.mean(axis=1)
    return SignalContract.clip_scores(pd.Series(score, index=frame.index, dtype=float))

