from __future__ import annotations

import numpy as np

from buffmini.ml.autoencoder import AutoencoderConfig, encode_windows, train_autoencoder
from buffmini.utils.hashing import stable_hash


def _windows() -> np.ndarray:
    rng = np.random.default_rng(123)
    base = rng.standard_normal((48, 24, 5), dtype=np.float32)
    drift = np.linspace(-0.2, 0.2, 24, dtype=np.float32)[None, :, None]
    return (base + drift).astype(np.float32)


def test_stage30_autoencoder_determinism_small_cpu() -> None:
    windows = _windows()
    cfg = AutoencoderConfig(embedding_dim=12, epochs=3, seed=9, device="cpu")
    model_a = train_autoencoder(windows, cfg=cfg, use_torch=False)
    model_b = train_autoencoder(windows, cfg=cfg, use_torch=False)
    emb_a = encode_windows(model_a, windows)
    emb_b = encode_windows(model_b, windows)
    assert stable_hash(emb_a.tolist(), length=16) == stable_hash(emb_b.tolist(), length=16)
    assert stable_hash(model_a["train_loss_history"], length=16) == stable_hash(model_b["train_loss_history"], length=16)

