"""Extract Stage-30 embeddings and store deterministic cache files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.ml.autoencoder import encode_windows, load_model
from buffmini.ml.embedding_cache import build_embedding_cache_frame
from buffmini.stage10.evaluate import _build_features
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Stage-30 embeddings")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--train-timeframe", type=str, default="15m")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--features-ml-dir", type=Path, default=Path("data/features_ml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _symbol_path(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def _windows_with_timestamps(
    frame: pd.DataFrame,
    *,
    window: int,
    stride: int,
) -> tuple[np.ndarray, list[pd.Timestamp]]:
    cols = ("open", "high", "low", "close", "volume")
    work = frame.copy().reset_index(drop=True)
    for col in cols:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    work = work.dropna(subset=list(cols)).reset_index(drop=True)
    ts = pd.to_datetime(work.get("timestamp"), utc=True, errors="coerce")
    vals = work.loc[:, list(cols)].to_numpy(dtype=np.float32)
    chunks: list[np.ndarray] = []
    out_ts: list[pd.Timestamp] = []
    start = 0
    n = int(work.shape[0])
    while start + int(window) <= n:
        end = int(start + window)
        chunks.append(vals[start:end, :].copy())
        out_ts.append(pd.Timestamp(ts.iloc[end - 1]).tz_convert("UTC"))
        start += int(stride)
    if not chunks:
        return np.empty((0, int(window), len(cols)), dtype=np.float32), []
    return np.stack(chunks, axis=0).astype(np.float32), out_ts


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    symbols = _csv(args.symbols)
    tf = str(args.train_timeframe).strip()
    model = load_model(Path(args.model_path))

    feature_map = _build_features(
        config=cfg,
        symbols=symbols,
        timeframe=tf,
        dry_run=bool(args.dry_run),
        seed=int(args.seed),
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    out_root = Path(args.features_ml_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    config_hash = compute_config_hash(cfg)
    snapshot = snapshot_metadata_from_config(cfg)

    cache_rows: list[dict[str, Any]] = []
    for symbol in sorted(feature_map.keys()):
        frame = feature_map[str(symbol)].copy().reset_index(drop=True)
        windows, ts_list = _windows_with_timestamps(
            frame,
            window=int(args.window),
            stride=int(args.stride),
        )
        if windows.shape[0] == 0:
            continue
        emb = encode_windows(model, windows)
        out = build_embedding_cache_frame(
            timestamps=ts_list,
            symbol=str(symbol),
            timeframe=tf,
            window=int(args.window),
            stride=int(args.stride),
            embeddings=emb,
        )
        sym_dir = out_root / _symbol_path(str(symbol))
        sym_dir.mkdir(parents=True, exist_ok=True)
        out_path = sym_dir / f"embeddings_{tf}.parquet"
        meta_path = sym_dir / f"embeddings_{tf}.meta.json"
        out.to_parquet(out_path, index=False)
        data_hash = stable_hash(out.to_dict(orient="list"), length=16)
        meta = {
            "stage": "30.3",
            "symbol": str(symbol),
            "timeframe": tf,
            "seed": int(args.seed),
            "window": int(args.window),
            "stride": int(args.stride),
            "rows": int(out.shape[0]),
            "embedding_dim": int(emb.shape[1]),
            "model_path": str(Path(args.model_path).as_posix()),
            "model_hash": stable_hash(
                {
                    "encoder_w": np.asarray(model.get("encoder_w", np.zeros(1, dtype=np.float32))).tolist(),
                    "encoder_b": np.asarray(model.get("encoder_b", np.zeros(1, dtype=np.float32))).tolist(),
                    "seed": int(model.get("seed", 0)),
                },
                length=16,
            ),
            "config_hash": config_hash,
            "data_hash": data_hash,
            "resolved_end_ts": pd.Timestamp(out["timestamp"].max()).isoformat(),
            **snapshot,
        }
        meta_path.write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
        cache_rows.append({"symbol": str(symbol), "parquet_path": str(out_path.as_posix()), "meta_path": str(meta_path.as_posix()), "rows": int(out.shape[0]), "data_hash": data_hash})

    payload = {
        "symbols": symbols,
        "timeframe": tf,
        "cache_rows": cache_rows,
    }
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
