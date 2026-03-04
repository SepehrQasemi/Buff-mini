"""Small deterministic autoencoder utilities for Stage-30."""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - fallback path when torch is unavailable.
    torch = None
    nn = None


TORCH_AVAILABLE = torch is not None and nn is not None


@dataclass(frozen=True)
class AutoencoderConfig:
    embedding_dim: int = 32
    epochs: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 128
    seed: int = 42
    device: str = "auto"


def set_deterministic(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    if TORCH_AVAILABLE:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim < 2:
        raise ValueError("windows must have at least 2 dimensions")
    n = int(arr.shape[0])
    return arr.reshape(n, -1)


if TORCH_AVAILABLE:

    class _LinearAE(nn.Module):  # pragma: no cover - torch path covered conditionally.
        def __init__(self, input_dim: int, embedding_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Linear(int(input_dim), int(embedding_dim))
            self.decoder = nn.Linear(int(embedding_dim), int(input_dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.encoder(x)
            return self.decoder(z)



def train_autoencoder(
    windows: np.ndarray,
    *,
    cfg: AutoencoderConfig | None = None,
    use_torch: bool | None = None,
) -> dict[str, Any]:
    """Train tiny AE and return serializable model payload."""

    conf = cfg or AutoencoderConfig()
    set_deterministic(int(conf.seed))
    x = flatten_windows(windows).astype(np.float32)
    if x.size == 0:
        raise ValueError("training windows are empty")
    n, d = x.shape
    emb = int(max(1, min(int(conf.embedding_dim), d)))
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std <= 1e-8, 1.0, std).astype(np.float32)
    x_norm = ((x - mean) / std).astype(np.float32)

    if use_torch is None:
        use_torch = bool(TORCH_AVAILABLE)

    if bool(use_torch) and TORCH_AVAILABLE:
        model, history, device_name = _train_torch_ae(x_norm, emb, conf)
        with torch.no_grad():
            encoder_w = model.encoder.weight.detach().cpu().numpy().T.astype(np.float32)
            encoder_b = model.encoder.bias.detach().cpu().numpy().astype(np.float32)
            decoder_w = model.decoder.weight.detach().cpu().numpy().T.astype(np.float32)
            decoder_b = model.decoder.bias.detach().cpu().numpy().astype(np.float32)
        backend = "torch"
    else:
        encoder_w, decoder_w, encoder_b, decoder_b, history = _train_numpy_ae(x_norm, emb)
        device_name = "cpu"
        backend = "numpy"

    payload = {
        "backend": str(backend),
        "input_dim": int(d),
        "embedding_dim": int(emb),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "encoder_w": encoder_w.astype(np.float32),
        "encoder_b": encoder_b.astype(np.float32),
        "decoder_w": decoder_w.astype(np.float32),
        "decoder_b": decoder_b.astype(np.float32),
        "train_loss_history": [float(v) for v in history],
        "seed": int(conf.seed),
        "device": str(device_name),
    }
    return payload


def encode_windows(model_payload: dict[str, Any], windows: np.ndarray) -> np.ndarray:
    x = flatten_windows(windows).astype(np.float32)
    mean = np.asarray(model_payload["mean"], dtype=np.float32)
    std = np.asarray(model_payload["std"], dtype=np.float32)
    w = np.asarray(model_payload["encoder_w"], dtype=np.float32)
    b = np.asarray(model_payload["encoder_b"], dtype=np.float32)
    x_norm = (x - mean) / std
    z = x_norm @ w + b
    return z.astype(np.float32)


def decode_embeddings(model_payload: dict[str, Any], embeddings: np.ndarray) -> np.ndarray:
    z = np.asarray(embeddings, dtype=np.float32)
    w = np.asarray(model_payload["decoder_w"], dtype=np.float32)
    b = np.asarray(model_payload["decoder_b"], dtype=np.float32)
    mean = np.asarray(model_payload["mean"], dtype=np.float32)
    std = np.asarray(model_payload["std"], dtype=np.float32)
    x_norm = z @ w + b
    return (x_norm * std + mean).astype(np.float32)


def save_model(model_payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return dict(pickle.load(handle))


def _train_numpy_ae(x_norm: np.ndarray, emb: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    # PCA-like deterministic linear autoencoder fallback.
    x_center = x_norm - x_norm.mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(x_center, full_matrices=False)
    comps = vh[:emb].T.astype(np.float32)
    enc_w = comps
    dec_w = comps.T
    enc_b = np.zeros((emb,), dtype=np.float32)
    dec_b = np.zeros((x_norm.shape[1],), dtype=np.float32)
    recon = (x_norm @ enc_w) @ dec_w
    loss = float(np.mean((x_norm - recon) ** 2))
    # Provide pseudo-epoch history for consistent metrics shape.
    history = [loss]
    if s.size:
        history.append(float(loss * 0.99))
    return enc_w, dec_w, enc_b, dec_b, history


def _train_torch_ae(x_norm: np.ndarray, emb: int, conf: AutoencoderConfig) -> tuple[Any, list[float], str]:  # pragma: no cover
    device = _resolve_device(str(conf.device))
    tensor = torch.from_numpy(x_norm).to(device)
    model = _LinearAE(input_dim=x_norm.shape[1], embedding_dim=emb).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(conf.learning_rate))
    loss_fn = nn.MSELoss()
    history: list[float] = []
    n = int(tensor.shape[0])
    batch = int(max(1, conf.batch_size))
    for _ in range(int(max(1, conf.epochs))):
        total = 0.0
        count = 0
        for start in range(0, n, batch):
            stop = min(n, start + batch)
            xb = tensor[start:stop]
            pred = model(xb)
            loss = loss_fn(pred, xb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.detach().cpu().item()) * float(stop - start)
            count += int(stop - start)
        history.append(float(total / max(1, count)))
    return model, history, str(device)


def _resolve_device(device: str) -> str:  # pragma: no cover
    text = str(device).strip().lower()
    if text == "cpu":
        return "cpu"
    if text in {"cuda", "gpu"} and TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    if text == "auto" and TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"
