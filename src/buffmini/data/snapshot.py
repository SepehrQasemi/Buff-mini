"""Deterministic data snapshot helpers for run reproducibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from buffmini.constants import CANONICAL_DATA_DIR
from buffmini.data.derived_tf import CANONICAL_TIMEFRAMES, canonical_meta_path
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def snapshot_metadata_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve snapshot id/hash from config-defined snapshot location."""

    data_cfg = dict(config.get("data", {}) or {})
    snapshot_cfg = dict(data_cfg.get("snapshot", {}) or {})
    snapshot_id = str(snapshot_cfg.get("id", "DATA_FROZEN_v1"))
    default_path = Path("data") / "snapshots" / f"{snapshot_id}.json"
    snapshot_path = Path(snapshot_cfg.get("path", default_path))
    if not snapshot_path.exists():
        return {
            "data_snapshot_id": snapshot_id,
            "data_snapshot_hash": "",
            "data_snapshot_path": str(snapshot_path.as_posix()),
            "data_snapshot_exists": False,
        }
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return {
        "data_snapshot_id": str(payload.get("snapshot_id", snapshot_id)),
        "data_snapshot_hash": str(payload.get("snapshot_hash", stable_hash(payload, length=16))),
        "data_snapshot_path": str(snapshot_path.as_posix()),
        "data_snapshot_exists": True,
    }


def build_snapshot_payload(
    *,
    snapshot_id: str,
    symbols: list[str],
    timeframes: list[str] | None = None,
    exchange: str = "binance",
    canonical_dir: Path = CANONICAL_DATA_DIR,
) -> dict[str, Any]:
    """Build deterministic snapshot payload from canonical timeframe meta files."""

    canonical_tfs = list(timeframes) if timeframes is not None else list(CANONICAL_TIMEFRAMES)
    per_symbol: dict[str, Any] = {}
    for symbol in symbols:
        tf_map: dict[str, Any] = {}
        for tf in canonical_tfs:
            meta_path = canonical_meta_path(
                symbol=str(symbol),
                timeframe=str(tf),
                exchange=str(exchange),
                canonical_dir=Path(canonical_dir),
            )
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            tf_map[str(tf)] = {
                "sha256": str(meta.get("sha256", "")),
                "start_ts": meta.get("start_ts"),
                "end_ts": meta.get("end_ts"),
                "candle_count": int(meta.get("candle_count", 0)),
            }
        per_symbol[str(symbol)] = tf_map
    payload = {
        "snapshot_id": str(snapshot_id),
        "created_at": utc_now_compact(),
        "exchange": str(exchange).strip().lower(),
        "symbols": list(symbols),
        "canonical_timeframes": canonical_tfs,
        "per_symbol_per_tf": per_symbol,
    }
    payload["snapshot_hash"] = stable_hash(payload, length=16)
    return payload


def write_snapshot_file(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path

