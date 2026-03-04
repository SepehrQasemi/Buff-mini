from __future__ import annotations

import numpy as np

from buffmini.ml.context_cluster import ClusterConfig, cluster_embeddings
from buffmini.utils.hashing import stable_hash


def test_stage30_context_clustering_reproducible() -> None:
    rng = np.random.default_rng(99)
    emb = rng.normal(size=(180, 16)).astype(np.float32)
    cfg = ClusterConfig(k=6, max_iters=60, seed=42)

    first = cluster_embeddings(embeddings=emb, cfg=cfg)
    second = cluster_embeddings(embeddings=emb, cfg=cfg)

    assert int(first["k"]) == 6
    assert int(second["k"]) == 6
    assert stable_hash(first["labels"].tolist(), length=16) == stable_hash(second["labels"].tolist(), length=16)
    assert stable_hash(first["centers"].tolist(), length=16) == stable_hash(second["centers"].tolist(), length=16)
    assert stable_hash(first["probs"].tolist(), length=16) == stable_hash(second["probs"].tolist(), length=16)

