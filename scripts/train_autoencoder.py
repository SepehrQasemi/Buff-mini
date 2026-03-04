"""Train a small deterministic autoencoder for Stage-30 embeddings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.ml.autoencoder import (
    AutoencoderConfig,
    encode_windows,
    save_model,
    train_autoencoder,
)
from buffmini.stage10.evaluate import _build_features
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-30 autoencoder")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--train-timeframe", type=str, default="15m")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _stack_windows(frames: dict[str, pd.DataFrame], *, window: int, stride: int) -> tuple[np.ndarray, dict[str, int], str | None]:
    cols = ("open", "high", "low", "close", "volume")
    chunks: list[np.ndarray] = []
    counts: dict[str, int] = {}
    resolved_end: list[pd.Timestamp] = []
    for symbol in sorted(frames.keys()):
        frame = frames[str(symbol)].reset_index(drop=True).copy()
        for col in cols:
            frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
        frame = frame.dropna(subset=list(cols)).reset_index(drop=True)
        ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
        if not ts.empty:
            resolved_end.append(pd.Timestamp(ts.max()).tz_convert("UTC"))
        n = int(frame.shape[0])
        local = 0
        start = 0
        values = frame.loc[:, list(cols)].to_numpy(dtype=np.float32)
        while start + int(window) <= n:
            stop = int(start + window)
            chunks.append(values[start:stop, :].copy())
            local += 1
            start += int(stride)
        counts[str(symbol)] = int(local)
    if not chunks:
        return np.empty((0, int(window), len(cols)), dtype=np.float32), counts, None
    out = np.stack(chunks, axis=0).astype(np.float32)
    resolved_end_ts = max(resolved_end).isoformat() if resolved_end else None
    return out, counts, resolved_end_ts


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    symbols = _csv(args.symbols)
    timeframe = str(args.train_timeframe).strip()

    features_by_symbol = _build_features(
        config=cfg,
        symbols=symbols,
        timeframe=timeframe,
        dry_run=bool(args.dry_run),
        seed=int(args.seed),
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    windows, counts, resolved_end_ts = _stack_windows(
        features_by_symbol,
        window=int(args.window),
        stride=int(args.stride),
    )
    if windows.shape[0] == 0:
        raise SystemExit("No windows generated for training.")

    conf = AutoencoderConfig(
        embedding_dim=int(args.embedding_dim),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=str(args.device),
    )
    model = train_autoencoder(windows, cfg=conf)
    embeddings = encode_windows(model, windows)

    config_hash = compute_config_hash(cfg)
    data_hash_parts: dict[str, str] = {}
    for symbol, frame in sorted(features_by_symbol.items()):
        cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in frame.columns]
        data_hash_parts[str(symbol)] = stable_hash(frame.loc[:, cols].to_dict(orient="list"), length=16) if cols else stable_hash({"rows": int(frame.shape[0])}, length=16)
    data_hash = stable_hash(data_hash_parts, length=16)

    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': int(args.seed), 'symbols': symbols, 'tf': timeframe, 'window': int(args.window), 'stride': int(args.stride), 'emb': int(args.embedding_dim), 'epochs': int(args.epochs), 'dry': bool(args.dry_run), 'cfg': config_hash, 'data': data_hash}, length=12)}"
        "_stage30_ae"
    )
    out_dir = Path(args.runs_dir) / run_id / "stage30"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pt"
    save_model(model, model_path)
    metrics: dict[str, Any] = {
        "stage": "30.2",
        "run_id": run_id,
        "seed": int(args.seed),
        "symbols": list(symbols),
        "train_timeframe": timeframe,
        "window": int(args.window),
        "stride": int(args.stride),
        "rows_total": int(windows.shape[0]),
        "rows_by_symbol": counts,
        "input_dim": int(model.get("input_dim", 0)),
        "embedding_dim": int(model.get("embedding_dim", 0)),
        "backend": str(model.get("backend", "unknown")),
        "device": str(model.get("device", "unknown")),
        "train_loss_first": float(model.get("train_loss_history", [0.0])[0]),
        "train_loss_last": float(model.get("train_loss_history", [0.0])[-1]),
        "embedding_mean_abs": float(np.mean(np.abs(embeddings))),
        "embedding_std": float(np.std(embeddings)),
        "model_hash": stable_hash(
            {
                "enc_w": np.asarray(model.get("encoder_w", np.zeros(1, dtype=np.float32))).tolist(),
                "enc_b": np.asarray(model.get("encoder_b", np.zeros(1, dtype=np.float32))).tolist(),
                "dec_w": np.asarray(model.get("decoder_w", np.zeros(1, dtype=np.float32))).tolist(),
                "dec_b": np.asarray(model.get("decoder_b", np.zeros(1, dtype=np.float32))).tolist(),
                "seed": int(model.get("seed", args.seed)),
            },
            length=16,
        ),
        "config_hash": config_hash,
        "data_hash": data_hash,
        "resolved_end_ts": resolved_end_ts,
        "runtime_seconds": float(time.perf_counter() - started),
        **snapshot_metadata_from_config(cfg),
    }
    metrics_path = out_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(f"run_id: {run_id}")
    print(f"model: {model_path}")
    print(f"train_metrics: {metrics_path}")


if __name__ == "__main__":
    main()

