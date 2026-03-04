from __future__ import annotations

import pandas as pd
import numpy as np

from buffmini.ml.embedding_cache import build_embedding_cache_frame
from buffmini.utils.hashing import stable_hash


def test_stage30_embedding_cache_contract() -> None:
    ts = pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC").tolist()
    rng = np.random.default_rng(5)
    emb = rng.normal(size=(20, 8)).astype(np.float32)

    first = build_embedding_cache_frame(
        timestamps=ts,
        symbol="BTC/USDT",
        timeframe="15m",
        window=256,
        stride=32,
        embeddings=emb,
    )
    second = build_embedding_cache_frame(
        timestamps=ts,
        symbol="BTC/USDT",
        timeframe="15m",
        window=256,
        stride=32,
        embeddings=emb,
    )

    assert "timestamp" in first.columns
    assert "symbol" in first.columns
    assert "timeframe" in first.columns
    emb_cols = [c for c in first.columns if c.startswith("emb_")]
    assert len(emb_cols) == 8
    assert first["symbol"].eq("BTC/USDT").all()
    assert first["timeframe"].eq("15m").all()
    assert stable_hash(first.to_dict(orient="records"), length=16) == stable_hash(second.to_dict(orient="records"), length=16)
