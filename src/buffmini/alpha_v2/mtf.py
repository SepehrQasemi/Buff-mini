"""Stage-22 MTF policies with strict causal alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


ConflictMode = Literal["net", "hedge", "isolated"]


@dataclass(frozen=True)
class MtfPolicyConfig:
    bias_threshold: float = 0.10
    entry_threshold: float = 0.25
    conflict_mode: ConflictMode = "net"


def causal_join_bias(
    *,
    base_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    base_ts_col: str = "timestamp",
    bias_ts_col: str = "timestamp",
    bias_col: str = "bias_score",
) -> pd.DataFrame:
    """Backward-only asof join to avoid lookahead."""

    left = base_df.sort_values(base_ts_col).copy()
    right = bias_df.sort_values(bias_ts_col).copy()
    left[base_ts_col] = pd.to_datetime(left[base_ts_col], utc=True, errors="coerce")
    right[bias_ts_col] = pd.to_datetime(right[bias_ts_col], utc=True, errors="coerce")
    out = pd.merge_asof(
        left,
        right[[bias_ts_col, bias_col]],
        left_on=base_ts_col,
        right_on=bias_ts_col,
        direction="backward",
    )
    out[bias_col] = pd.to_numeric(out[bias_col], errors="coerce").fillna(0.0)
    assert bool((pd.to_datetime(out[bias_ts_col], utc=True, errors="coerce") <= out[base_ts_col]).fillna(True).all())
    return out


def apply_mtf_policy(
    *,
    base_df: pd.DataFrame,
    entry_score_col: str,
    bias_score_col: str,
    cfg: MtfPolicyConfig,
) -> tuple[pd.Series, dict[str, float]]:
    """Apply bias/entry conflict resolution without leakage."""

    entry = pd.to_numeric(base_df.get(entry_score_col, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    bias = pd.to_numeric(base_df.get(bias_score_col, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    entry_side = np.where(entry >= cfg.entry_threshold, 1, np.where(entry <= -cfg.entry_threshold, -1, 0))
    bias_side = np.where(bias >= cfg.bias_threshold, 1, np.where(bias <= -cfg.bias_threshold, -1, 0))
    disagree = (entry_side != 0) & (bias_side != 0) & (entry_side != bias_side)

    if cfg.conflict_mode == "net":
        final = np.where(disagree, 0, entry_side)
    elif cfg.conflict_mode == "hedge":
        final = entry_side  # keep entry; hedge handled upstream via exposure rules.
    else:  # isolated
        final = np.where(bias_side == 0, entry_side, np.where(disagree, 0, entry_side))

    stats = {
        "conflict_rate_pct": float(np.mean(disagree) * 100.0),
        "bias_alignment_rate_pct": float(np.mean((entry_side == bias_side) | (entry_side == 0) | (bias_side == 0)) * 100.0),
        "nonzero_signal_pct": float(np.mean(final != 0) * 100.0),
    }
    return pd.Series(final, index=base_df.index, dtype=int), stats

