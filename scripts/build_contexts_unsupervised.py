"""Build unsupervised context labels from cached Stage-30 embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.ml.context_cluster import ClusterConfig, cluster_embeddings, labels_to_frame
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-30 unsupervised contexts")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--features-ml-dir", type=Path, default=Path("data/features_ml"))
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _symbol_path(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    symbols = _csv(args.symbols)
    timeframe = str(args.timeframe).strip()

    labels_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for symbol in symbols:
        path = Path(args.features_ml_dir) / _symbol_path(symbol) / f"embeddings_{timeframe}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        emb_cols = [c for c in frame.columns if str(c).startswith("emb_")]
        if not emb_cols:
            continue
        emb = frame.loc[:, emb_cols].to_numpy(dtype="float32")
        clustered = cluster_embeddings(
            emb,
            cfg=ClusterConfig(k=int(args.k), seed=int(args.seed)),
        )
        labels = labels_to_frame(
            timestamps=pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce"),
            symbol=str(symbol),
            timeframe=str(timeframe),
            labels=clustered["labels"],
            probs=clustered["probs"],
        )
        labels_rows.append(labels)
        dist = labels["context_label"].value_counts(normalize=True).sort_index().to_dict()
        summaries.append(
            {
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "rows": int(labels.shape[0]),
                "k": int(clustered["k"]),
                "inertia": float(clustered["inertia"]),
                "context_distribution": {str(k): float(v) for k, v in dist.items()},
            }
        )
        hashes[str(symbol)] = stable_hash(frame.loc[:, ["timestamp", *emb_cols]].to_dict(orient="list"), length=16)

    all_labels = pd.concat(labels_rows, ignore_index=True) if labels_rows else pd.DataFrame()
    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': int(args.seed), 'symbols': symbols, 'tf': timeframe, 'k': int(args.k), 'cfg': compute_config_hash(cfg), 'data': hashes}, length=12)}"
        "_stage30_ctx"
    )
    out_dir = Path(args.runs_dir) / run_id / "stage30"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "context_labels.parquet"
    json_path = out_dir / "context_summary.json"
    all_labels.to_parquet(out_path, index=False)

    payload: dict[str, Any] = {
        "stage": "30.3",
        "run_id": run_id,
        "seed": int(args.seed),
        "symbols": symbols,
        "timeframe": timeframe,
        "k": int(args.k),
        "rows_total": int(all_labels.shape[0]),
        "summary_by_symbol": summaries,
        "config_hash": compute_config_hash(cfg),
        "data_hash": stable_hash(hashes, length=16),
        "resolved_end_ts": (
            pd.Timestamp(all_labels["timestamp"].max()).isoformat()
            if not all_labels.empty and "timestamp" in all_labels.columns
            else None
        ),
        **snapshot_metadata_from_config(cfg),
    }
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"run_id: {run_id}")
    print(f"context_labels: {out_path}")
    print(f"context_summary: {json_path}")


if __name__ == "__main__":
    main()

