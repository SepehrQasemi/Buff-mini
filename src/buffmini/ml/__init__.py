"""Machine-learning utilities for offline representation learning."""

from buffmini.ml.autoencoder import (
    AutoencoderConfig,
    decode_embeddings,
    encode_windows,
    load_model,
    save_model,
    set_deterministic,
    train_autoencoder,
)
from buffmini.ml.dataset import build_dataset_index

__all__ = [
    "AutoencoderConfig",
    "build_dataset_index",
    "decode_embeddings",
    "encode_windows",
    "load_model",
    "save_model",
    "set_deterministic",
    "train_autoencoder",
]
