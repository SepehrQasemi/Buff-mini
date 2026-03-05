"""Stage-34 model registry for evolutionary memory across generations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.utils.hashing import stable_hash


@dataclass(frozen=True)
class RegistryEntry:
    model_id: str
    generation: int
    seed: int
    symbol: str
    timeframe: str
    horizon: str
    feature_subset_sig: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any]
    data_hash: str
    resolved_end_ts: str | None
    parent_model_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": str(self.model_id),
            "generation": int(self.generation),
            "seed": int(self.seed),
            "symbol": str(self.symbol),
            "timeframe": str(self.timeframe),
            "horizon": str(self.horizon),
            "feature_subset_sig": str(self.feature_subset_sig),
            "hyperparameters": dict(self.hyperparameters),
            "metrics": dict(self.metrics),
            "data_hash": str(self.data_hash),
            "resolved_end_ts": self.resolved_end_ts,
            "parent_model_ids": list(self.parent_model_ids),
        }


def registry_model_id(payload: dict[str, Any]) -> str:
    return f"m_{stable_hash(payload, length=16)}"


def load_registry(path: Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for row in raw:
        if isinstance(row, dict) and str(row.get("model_id", "")).strip():
            out.append(_normalize_row(dict(row)))
    return sorted(out, key=lambda r: (int(r.get("generation", 0)), str(r.get("model_id", ""))))


def save_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    normalized = [_normalize_row(dict(r)) for r in rows if isinstance(r, dict)]
    normalized = sorted(normalized, key=lambda r: (int(r.get("generation", 0)), str(r.get("model_id", ""))))
    p.write_text(json.dumps(normalized, indent=2, allow_nan=False), encoding="utf-8")


def upsert_entry(path: Path, entry: RegistryEntry | dict[str, Any]) -> list[dict[str, Any]]:
    row = entry.as_dict() if isinstance(entry, RegistryEntry) else _normalize_row(dict(entry))
    rows = load_registry(path)
    model_id = str(row.get("model_id", "")).strip() or registry_model_id(row)
    row["model_id"] = model_id
    updated = [r for r in rows if str(r.get("model_id", "")) != model_id]
    updated.append(row)
    save_registry(path, updated)
    return load_registry(path)


def top_models(path: Path, *, top_k: int = 5) -> pd.DataFrame:
    rows = load_registry(path)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["exp_lcb"] = pd.to_numeric(frame["metrics"].map(lambda m: (m or {}).get("exp_lcb", 0.0)), errors="coerce").fillna(0.0)
    frame["stability"] = pd.to_numeric(frame["metrics"].map(lambda m: (m or {}).get("positive_windows_ratio", 0.0)), errors="coerce").fillna(0.0)
    frame["drawdown"] = pd.to_numeric(frame["metrics"].map(lambda m: (m or {}).get("maxdd_p95", 1.0)), errors="coerce").fillna(1.0)
    frame = frame.sort_values(["exp_lcb", "stability", "drawdown", "generation", "model_id"], ascending=[False, False, True, False, True])
    return frame.head(int(max(1, top_k))).reset_index(drop=True)


def select_elites(path: Path, *, generation: int, elite_count: int) -> list[dict[str, Any]]:
    frame = top_models(path, top_k=max(1, int(elite_count) * 3))
    if frame.empty:
        return []
    frame = frame.loc[frame["generation"] <= int(generation)].copy()
    if frame.empty:
        return []
    out = frame.head(int(max(1, elite_count)))
    return [dict(v) for v in out.to_dict(orient="records")]


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["model_id"] = str(out.get("model_id", "")).strip() or registry_model_id(out)
    out["generation"] = int(out.get("generation", 0))
    out["seed"] = int(out.get("seed", 0))
    out["symbol"] = str(out.get("symbol", ""))
    out["timeframe"] = str(out.get("timeframe", ""))
    out["horizon"] = str(out.get("horizon", ""))
    out["feature_subset_sig"] = str(out.get("feature_subset_sig", ""))
    out["hyperparameters"] = dict(out.get("hyperparameters", {}) or {})
    out["metrics"] = dict(out.get("metrics", {}) or {})
    out["data_hash"] = str(out.get("data_hash", ""))
    out["resolved_end_ts"] = out.get("resolved_end_ts")
    out["parent_model_ids"] = [str(v) for v in (out.get("parent_model_ids", []) or [])]
    return out
