"""Embedding cache frame utilities for Stage-30."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_embedding_cache_frame(
    *,
    timestamps: list[pd.Timestamp],
    symbol: str,
    timeframe: str,
    window: int,
    stride: int,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    emb = np.asarray(embeddings, dtype=np.float32)
    n = int(min(len(timestamps), emb.shape[0]))
    emb_cols = {f"emb_{idx}": emb[:n, idx].astype(np.float32) for idx in range(int(emb.shape[1]))}
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps[:n], utc=True, errors="coerce"),
            "symbol": str(symbol),
            "timeframe": str(timeframe),
            "window": int(window),
            "stride": int(stride),
            **emb_cols,
        }
    )

