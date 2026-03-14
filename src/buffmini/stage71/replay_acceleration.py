"""Stage-71 replay acceleration measurement."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from buffmini.stage55 import build_replay_cache_key
from buffmini.utils.hashing import stable_hash


def _baseline_loop(candidates: pd.DataFrame, returns: np.ndarray) -> float:
    total = 0.0
    r = np.asarray(returns, dtype=float)
    n = int(len(r))
    for rec in candidates.to_dict(orient="records"):
        seed = int(_stable_seed(str(rec.get("candidate_id", "")), n))
        pnl = 0.0
        for idx in range(seed, min(seed + 64, n)):
            pnl += float(r[idx]) * 0.97
        total += pnl
    return float(total)


def _vectorized_path(candidates: pd.DataFrame, returns: np.ndarray) -> float:
    n = int(len(returns))
    ids = candidates.get("candidate_id", pd.Series(dtype=str)).astype(str).tolist()
    seeds = np.asarray([int(_stable_seed(cid, n)) for cid in ids], dtype=int)
    window = 64
    r = np.asarray(returns, dtype=float)
    prefix = np.concatenate([np.asarray([0.0], dtype=float), np.cumsum(r)])
    start_idx = np.clip(seeds, 0, n)
    end_idx = np.clip(seeds + window, 0, n)
    pnl = (prefix[end_idx] - prefix[start_idx]) * 0.97
    return float(np.sum(pnl))


def _stable_seed(text: str, modulus: int) -> int:
    if int(modulus) <= 0:
        return 0
    digest = stable_hash({"candidate_id": str(text)}, length=16)
    return int(str(digest)[:12], 16) % int(modulus)


def measure_replay_acceleration(
    *,
    candidates: pd.DataFrame,
    returns: np.ndarray,
    data_hash: str,
    setup_signature: str,
    timeframe: str,
    cost_model: str,
    scope_id: str,
) -> dict[str, Any]:
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return {
            "status": "PARTIAL",
            "blocker_reason": "empty_candidates",
            "speedup_pct": 0.0,
        }
    cache_key = build_replay_cache_key(
        data_hash=str(data_hash),
        setup_signature=str(setup_signature),
        timeframe=str(timeframe),
        cost_model=str(cost_model),
        scope_id=str(scope_id),
    )
    t0 = time.perf_counter()
    baseline_total = _baseline_loop(frame, returns)
    baseline_seconds = float(time.perf_counter() - t0)

    t1 = time.perf_counter()
    optimized_total = _vectorized_path(frame, returns)
    optimized_seconds = float(time.perf_counter() - t1)

    speedup_pct = float(((baseline_seconds - optimized_seconds) / max(baseline_seconds, 1e-9)) * 100.0)
    return {
        "status": "SUCCESS",
        "cache_key": cache_key,
        "baseline_runtime_seconds": float(round(baseline_seconds, 8)),
        "optimized_runtime_seconds": float(round(optimized_seconds, 8)),
        "speedup_pct": float(round(speedup_pct, 6)),
        "meets_target_40pct": bool(speedup_pct >= 40.0),
        "baseline_total": float(round(baseline_total, 10)),
        "optimized_total": float(round(optimized_total, 10)),
        "consistency_delta": float(round(abs(baseline_total - optimized_total), 10)),
        "blocker_reason": "",
    }
