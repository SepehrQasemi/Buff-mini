"""Build deterministic Stage-30 dataset index from local cached data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.ml.dataset import build_dataset_index, infer_resolved_end_ts
from buffmini.stage10.evaluate import _build_features
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-30 dataset index")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--base-timeframe", type=str, default="1m")
    parser.add_argument("--train-timeframe", type=str, default="15m")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _frame_data_hash(frame: pd.DataFrame) -> str:
    cols = [col for col in ("timestamp", "open", "high", "low", "close", "volume") if col in frame.columns]
    if not cols:
        return stable_hash({"rows": int(frame.shape[0])}, length=16)
    return stable_hash(frame.loc[:, cols].to_dict(orient="list"), length=16)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    symbols = _csv(args.symbols)
    timeframe = str(args.train_timeframe).strip()
    features_by_symbol = _build_features(
        config=config,
        symbols=symbols,
        timeframe=timeframe,
        dry_run=bool(args.dry_run),
        seed=int(args.seed),
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    index = build_dataset_index(
        frames_by_symbol=features_by_symbol,
        timeframe=timeframe,
        window=int(args.window),
        stride=int(args.stride),
    )
    data_hash_map = {str(sym): _frame_data_hash(frame) for sym, frame in sorted(features_by_symbol.items())}
    data_hash = stable_hash(data_hash_map, length=16)
    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': int(args.seed), 'symbols': symbols, 'tf': timeframe, 'window': int(args.window), 'stride': int(args.stride), 'dry': bool(args.dry_run), 'cfg': compute_config_hash(config), 'data': data_hash}, length=12)}"
        "_stage30_ds"
    )
    out_dir = Path(args.runs_dir) / run_id / "stage30"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "dataset_index.parquet"
    meta_path = out_dir / "dataset_meta.json"
    index.to_parquet(index_path, index=False)
    payload: dict[str, Any] = {
        "stage": "30.1",
        "run_id": run_id,
        "seed": int(args.seed),
        "symbols": symbols,
        "base_timeframe": str(args.base_timeframe),
        "train_timeframe": timeframe,
        "window": int(args.window),
        "stride": int(args.stride),
        "dry_run": bool(args.dry_run),
        "rows_total": int(index.shape[0]),
        "rows_by_symbol": {
            str(sym): int((index.get("symbol", pd.Series(dtype=str)).astype(str) == str(sym)).sum())
            for sym in symbols
        },
        "dataset_hash": stable_hash(index.to_dict(orient="records"), length=16) if not index.empty else stable_hash("empty_dataset", length=16),
        "config_hash": compute_config_hash(config),
        "data_hash": data_hash,
        "resolved_end_ts": infer_resolved_end_ts(features_by_symbol),
        **snapshot_metadata_from_config(config),
    }
    meta_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"run_id: {run_id}")
    print(f"dataset_index: {index_path}")
    print(f"dataset_meta: {meta_path}")


if __name__ == "__main__":
    main()

