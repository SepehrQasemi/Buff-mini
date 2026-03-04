"""Context-aware WF/MC usability logic for Stage-28."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UsabilityConfig:
    min_trades_context: int = 30
    min_occurrences_context: int = 50
    min_windows: int = 3
    rare_pool_min_trades: int = 30


def compute_usability(
    *,
    candidate: dict[str, Any],
    windows: pd.DataFrame,
    cfg: UsabilityConfig | None = None,
) -> dict[str, Any]:
    """Compute deterministic context-aware usability outcome."""

    conf = cfg or UsabilityConfig()
    occurrences = int(max(0, candidate.get("context_occurrences", candidate.get("occurrences", 0)) or 0))
    trades = int(max(0, candidate.get("trades_in_context", candidate.get("trade_count", 0)) or 0))
    windows_df = windows.copy() if isinstance(windows, pd.DataFrame) else pd.DataFrame()
    if windows_df.empty:
        windows_df = pd.DataFrame(columns=["window_id", "trade_count", "occurrences"])
    window_count = int(windows_df.shape[0])
    window_trades = pd.to_numeric(windows_df.get("trade_count", 0), errors="coerce").fillna(0.0)
    window_occ = pd.to_numeric(
        windows_df.get("occurrences", windows_df.get("context_occurrences", 0)),
        errors="coerce",
    ).fillna(0.0)
    pooled_trades = int(window_trades.sum())
    pooled_occurrences = int(window_occ.sum()) if not window_occ.empty else int(occurrences)

    if occurrences < int(conf.min_occurrences_context):
        return _payload(
            usable=False,
            reason="insufficient_occurrences_context",
            occurrences=occurrences,
            trades=trades,
            windows=window_count,
            pooled_trades=pooled_trades,
            pooled_occurrences=pooled_occurrences,
            wf_triggered=False,
            mc_triggered=False,
            mc_pooling=False,
        )

    if trades >= int(conf.min_trades_context):
        wf_ok = bool(window_count >= int(conf.min_windows))
        return _payload(
            usable=True,
            reason="direct_context_usable",
            occurrences=occurrences,
            trades=trades,
            windows=window_count,
            pooled_trades=pooled_trades,
            pooled_occurrences=pooled_occurrences,
            wf_triggered=wf_ok,
            mc_triggered=wf_ok,
            mc_pooling=False,
        )

    pooled_ok = bool(
        pooled_occurrences >= int(conf.min_occurrences_context)
        and pooled_trades >= int(conf.rare_pool_min_trades)
    )
    if pooled_ok:
        wf_ok = bool(window_count >= int(conf.min_windows))
        return _payload(
            usable=True,
            reason="rare_context_pooled",
            occurrences=occurrences,
            trades=trades,
            windows=window_count,
            pooled_trades=pooled_trades,
            pooled_occurrences=pooled_occurrences,
            wf_triggered=wf_ok,
            mc_triggered=wf_ok,
            mc_pooling=True,
        )

    return _payload(
        usable=False,
        reason="insufficient_trades_context",
        occurrences=occurrences,
        trades=trades,
        windows=window_count,
        pooled_trades=pooled_trades,
        pooled_occurrences=pooled_occurrences,
        wf_triggered=False,
        mc_triggered=False,
        mc_pooling=False,
    )


def pool_returns_for_mc(
    *,
    windows_returns: list[np.ndarray | list[float]],
    min_total_trades: int = 30,
) -> tuple[np.ndarray, bool]:
    """Pool sparse per-window returns for rare-context MC validation."""

    arrays: list[np.ndarray] = []
    for item in windows_returns:
        arr = np.asarray(item, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            arrays.append(arr)
    if not arrays:
        return np.asarray([], dtype=float), False
    pooled = np.concatenate(arrays, axis=0)
    ok = bool(pooled.size >= int(min_total_trades))
    return pooled, ok


def _payload(
    *,
    usable: bool,
    reason: str,
    occurrences: int,
    trades: int,
    windows: int,
    pooled_trades: int,
    pooled_occurrences: int,
    wf_triggered: bool,
    mc_triggered: bool,
    mc_pooling: bool,
) -> dict[str, Any]:
    return {
        "usable": bool(usable),
        "reason": str(reason),
        "context_occurrences": int(occurrences),
        "trades_in_context": int(trades),
        "windows_count": int(windows),
        "pooled_trades": int(pooled_trades),
        "pooled_occurrences": int(pooled_occurrences),
        "wf_triggered": bool(wf_triggered),
        "mc_triggered": bool(mc_triggered),
        "mc_pooling": bool(mc_pooling),
    }

