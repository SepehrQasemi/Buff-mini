from __future__ import annotations

import numpy as np

from buffmini.ml.autoencoder import AutoencoderConfig, decode_embeddings, encode_windows, train_autoencoder


def _windows(n: int = 64, w: int = 32, f: int = 5) -> np.ndarray:
    rng = np.random.default_rng(7)
    arr = rng.normal(loc=0.0, scale=1.0, size=(n, w, f)).astype(np.float32)
    trend = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    return arr + trend


def test_stage30_autoencoder_shapes() -> None:
    windows = _windows()
    model = train_autoencoder(
        windows,
        cfg=AutoencoderConfig(embedding_dim=16, epochs=2, seed=42),
        use_torch=False,
    )
    emb = encode_windows(model, windows)
    rec = decode_embeddings(model, emb)

    assert model["input_dim"] == int(windows.shape[1] * windows.shape[2])
    assert emb.shape == (windows.shape[0], 16)
    assert rec.shape == (windows.shape[0], model["input_dim"])
    assert np.isfinite(emb).all()
    assert np.isfinite(rec).all()

