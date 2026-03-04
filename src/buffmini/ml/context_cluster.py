"""Deterministic unsupervised context clustering for Stage-30."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusterConfig:
    k: int = 8
    max_iters: int = 100
    tol: float = 1e-6
    seed: int = 42


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    cfg: ClusterConfig | None = None,
) -> dict[str, Any]:
    """Run deterministic k-means style clustering on embedding vectors."""

    conf = cfg or ClusterConfig()
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be 2D")
    n, d = x.shape
    if n == 0 or d == 0:
        raise ValueError("embeddings cannot be empty")
    k = int(max(1, min(int(conf.k), n)))
    rng = np.random.default_rng(int(conf.seed))
    centers = _kmeans_plus_plus_init(x, k=k, rng=rng)
    labels = np.zeros(n, dtype=np.int64)
    inertia_prev = float("inf")

    for _ in range(int(max(1, conf.max_iters))):
        d2 = _squared_distances(x, centers)
        labels = np.argmin(d2, axis=1).astype(np.int64)
        new_centers = np.zeros_like(centers)
        for idx in range(k):
            mask = labels == idx
            if not np.any(mask):
                # Deterministic reseed of empty cluster to the point farthest from all centers.
                farthest = int(np.argmax(np.min(d2, axis=1)))
                new_centers[idx] = x[farthest]
            else:
                new_centers[idx] = x[mask].mean(axis=0)
        inertia = float(np.min(d2, axis=1).sum())
        shift = float(np.linalg.norm(new_centers - centers))
        centers = new_centers
        if abs(inertia_prev - inertia) <= float(conf.tol) and shift <= float(conf.tol):
            break
        inertia_prev = inertia

    final_d2 = _squared_distances(x, centers)
    labels = np.argmin(final_d2, axis=1).astype(np.int64)
    probs = _distance_softmax(final_d2)
    return {
        "labels": labels.astype(np.int64),
        "centers": centers.astype(np.float32),
        "probs": probs.astype(np.float32),
        "inertia": float(np.min(final_d2, axis=1).sum()),
        "k": int(k),
    }


def labels_to_frame(
    *,
    timestamps: pd.Series,
    symbol: str,
    timeframe: str,
    labels: np.ndarray,
    probs: np.ndarray,
) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    n = min(len(ts), int(labels.shape[0]), int(probs.shape[0]))
    out = pd.DataFrame(
        {
            "timestamp": ts.iloc[:n].reset_index(drop=True),
            "symbol": str(symbol),
            "timeframe": str(timeframe),
            "context_label": [f"CTX_{int(v)}" for v in labels[:n]],
            "context_id": labels[:n].astype(int),
        }
    )
    for idx in range(int(probs.shape[1])):
        out[f"context_prob_{idx}"] = probs[:n, idx].astype(np.float32)
    return out


def _kmeans_plus_plus_init(x: np.ndarray, *, k: int, rng: np.random.Generator) -> np.ndarray:
    n = int(x.shape[0])
    first = int(rng.integers(0, n))
    centers = [x[first]]
    while len(centers) < int(k):
        d2 = _squared_distances(x, np.asarray(centers))
        closest = np.min(d2, axis=1)
        total = float(np.sum(closest))
        if total <= 0.0 or not np.isfinite(total):
            idx = int(rng.integers(0, n))
            centers.append(x[idx])
            continue
        probs = closest / total
        idx = int(rng.choice(np.arange(n), p=probs))
        centers.append(x[idx])
    return np.asarray(centers, dtype=np.float64)


def _squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)


def _distance_softmax(d2: np.ndarray) -> np.ndarray:
    # Convert distances to pseudo-probabilities (smaller distance => larger probability).
    logits = -np.asarray(d2, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return exp / denom

