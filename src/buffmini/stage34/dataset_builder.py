"""Stage-34 supervised dataset builder with deterministic row policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import CANONICAL_DATA_DIR, DERIVED_DATA_DIR
from buffmini.data.derived_tf import get_timeframe
from buffmini.stage34.features import compute_ohlcv_features, feature_columns
from buffmini.stage34.labels import build_labels, label_columns
from buffmini.utils.hashing import stable_hash


@dataclass(frozen=True)
class DatasetConfig:
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    timeframes: tuple[str, ...] = ("15m", "30m", "1h", "4h")
    max_rows_per_symbol: int = 300000
    max_features: int = 120
    horizons_hours: tuple[int, ...] = (24, 72)
    resolved_end_ts: str | None = None
    exchange: str = "binance"


def build_stage34_dataset(
    *,
    cfg: DatasetConfig,
    canonical_dir: Path = CANONICAL_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    data_hash_parts: dict[str, str] = {}
    feat_cols = feature_columns(max_features=int(cfg.max_features))
    lbl_cols = label_columns()
    resolved_end = pd.to_datetime(cfg.resolved_end_ts, utc=True, errors="coerce") if cfg.resolved_end_ts else None

    row_counts: dict[str, int] = {}
    for symbol in [str(v) for v in cfg.symbols]:
        for timeframe in [str(v) for v in cfg.timeframes]:
            loaded = get_timeframe(
                symbol=str(symbol),
                timeframe=str(timeframe),
                exchange=str(cfg.exchange),
                canonical_dir=Path(canonical_dir),
                derived_dir=Path(derived_dir),
                drop_incomplete_last=True,
                generator_version="stage34",
            )
            frame = loaded.frame.copy().sort_values("timestamp").reset_index(drop=True)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            if resolved_end is not None and pd.notna(resolved_end):
                frame = frame.loc[frame["timestamp"] <= resolved_end].reset_index(drop=True)
            feats = compute_ohlcv_features(frame)
            labels = build_labels(frame.assign(atr_pct=feats.get("atr_pct", 0.0)), timeframe=str(timeframe), horizons_hours=list(cfg.horizons_hours))
            merged = (
                frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]]
                .merge(feats, on="timestamp", how="left", suffixes=("", "_feat"))
                .merge(labels, on="timestamp", how="left", suffixes=("", "_lbl"))
            )
            keep = ["timestamp", "open", "high", "low", "close", "volume", *feat_cols, *lbl_cols]
            keep = [c for c in keep if c in merged.columns]
            merged = merged.loc[:, keep].copy()
            merged["symbol"] = str(symbol)
            merged["timeframe"] = str(timeframe)
            merged = merged.replace([float("inf"), float("-inf")], pd.NA)
            merged = merged.dropna(subset=[*feat_cols, *lbl_cols], how="any")
            if int(cfg.max_rows_per_symbol) > 0 and merged.shape[0] > int(cfg.max_rows_per_symbol):
                merged = merged.tail(int(cfg.max_rows_per_symbol)).reset_index(drop=True)
            row_counts[f"{symbol}|{timeframe}"] = int(merged.shape[0])
            if not merged.empty:
                rows.append(merged)
            source_cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in frame.columns]
            data_hash_parts[f"{symbol}|{timeframe}"] = stable_hash(frame.loc[:, source_cols].to_dict(orient="list"), length=16)

    dataset = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not dataset.empty:
        dataset = dataset.sort_values(["timestamp", "symbol", "timeframe"]).reset_index(drop=True)
    meta = {
        "symbols": list(cfg.symbols),
        "timeframes": list(cfg.timeframes),
        "row_counts": row_counts,
        "rows_total": int(dataset.shape[0]),
        "feature_columns": list(feat_cols),
        "label_columns": list(lbl_cols),
        "horizons_hours": [int(v) for v in cfg.horizons_hours],
        "max_rows_per_symbol": int(cfg.max_rows_per_symbol),
        "max_features": int(cfg.max_features),
        "resolved_end_ts": cfg.resolved_end_ts,
        "data_hash": stable_hash(data_hash_parts, length=16),
        "dataset_hash": stable_hash(dataset.to_dict(orient="list"), length=16) if not dataset.empty else stable_hash({"empty": True}, length=16),
    }
    return dataset, meta


def write_stage34_dataset(
    *,
    dataset: pd.DataFrame,
    meta: dict[str, Any],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Persist dataset + meta in deterministic format."""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    dataset_path = target / "dataset.parquet"
    meta_path = target / "dataset_meta.json"
    dataset.to_parquet(dataset_path, index=False, compression="zstd")
    meta_path.write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    return dataset_path, meta_path
