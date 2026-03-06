"""Stage-37 failure-aware self-learning helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LearningRegistryEntry:
    """Failure-aware model/family memory entry."""

    run_id: str
    generation: int
    family: str
    feature_subset_signature: str
    threshold_configuration: dict[str, Any]
    raw_signal_count: int
    activation_rate: float
    top_reject_reason: str
    cost_gate_fail_rate: float
    feasibility_fail_rate: float
    final_trade_count: int
    exp_lcb: float
    stability_score: float
    status: str = "active"
    elite: bool = False
    failure_motif_tags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": str(self.run_id),
            "generation": int(self.generation),
            "family": str(self.family),
            "feature_subset_signature": str(self.feature_subset_signature),
            "threshold_configuration": dict(self.threshold_configuration),
            "raw_signal_count": int(self.raw_signal_count),
            "activation_rate": float(self.activation_rate),
            "top_reject_reason": str(self.top_reject_reason),
            "cost_gate_fail_rate": float(self.cost_gate_fail_rate),
            "feasibility_fail_rate": float(self.feasibility_fail_rate),
            "final_trade_count": int(self.final_trade_count),
            "exp_lcb": float(self.exp_lcb),
            "stability_score": float(self.stability_score),
            "status": str(self.status),
            "elite": bool(self.elite),
            "failure_motif_tags": [str(v) for v in self.failure_motif_tags],
        }


def load_learning_registry(path: Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for row in raw:
        if isinstance(row, dict):
            out.append(_normalize_row(dict(row)))
    return sorted(out, key=lambda row: (int(row.get("generation", 0)), str(row.get("run_id", "")), str(row.get("family", ""))))


def save_learning_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    normalized = [_normalize_row(dict(row)) for row in rows if isinstance(row, dict)]
    normalized = sorted(normalized, key=lambda row: (int(row.get("generation", 0)), str(row.get("run_id", "")), str(row.get("family", ""))))
    p.write_text(json.dumps(normalized, indent=2, allow_nan=False), encoding="utf-8")


def upsert_learning_registry_entry(path: Path, entry: LearningRegistryEntry | dict[str, Any]) -> list[dict[str, Any]]:
    row = entry.as_dict() if isinstance(entry, LearningRegistryEntry) else _normalize_row(dict(entry))
    rows = load_learning_registry(path)
    key = (
        str(row.get("run_id", "")),
        int(row.get("generation", 0)),
        str(row.get("family", "")),
        str(row.get("feature_subset_signature", "")),
    )
    updated = [
        item
        for item in rows
        if (
            str(item.get("run_id", "")),
            int(item.get("generation", 0)),
            str(item.get("family", "")),
            str(item.get("feature_subset_signature", "")),
        )
        != key
    ]
    updated.append(row)
    save_learning_registry(path, updated)
    return load_learning_registry(path)


def apply_elite_flags(rows: list[dict[str, Any]], elite_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark elites deterministically in registry rows."""

    elite_keys = {
        (
            str(row.get("run_id", "")),
            int(row.get("generation", 0)),
            str(row.get("family", "")),
            str(row.get("feature_subset_signature", "")),
        )
        for row in elite_rows
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        item = _normalize_row(dict(row))
        key = (
            str(item.get("run_id", "")),
            int(item.get("generation", 0)),
            str(item.get("family", "")),
            str(item.get("feature_subset_signature", "")),
        )
        item["elite"] = bool(key in elite_keys)
        out.append(item)
    return sorted(
        out,
        key=lambda row: (
            int(row.get("generation", 0)),
            str(row.get("run_id", "")),
            str(row.get("family", "")),
            str(row.get("feature_subset_signature", "")),
        ),
    )


def select_elites_deterministic(rows: list[dict[str, Any]], *, top_k: int = 5) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    frame["exp_lcb"] = pd.to_numeric(frame.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
    frame["activation_rate"] = pd.to_numeric(frame.get("activation_rate", 0.0), errors="coerce").fillna(0.0)
    frame["stability_score"] = pd.to_numeric(frame.get("stability_score", 0.0), errors="coerce").fillna(0.0)
    frame["final_trade_count"] = pd.to_numeric(frame.get("final_trade_count", 0), errors="coerce").fillna(0).astype(int)
    frame = frame.sort_values(
        ["exp_lcb", "activation_rate", "stability_score", "final_trade_count", "family", "run_id"],
        ascending=[False, False, False, False, True, True],
    )
    return [dict(row) for row in frame.head(int(max(1, top_k))).to_dict(orient="records")]


def compute_family_exploration_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Down-weight dead-end families and keep successful families searchable."""

    frame = pd.DataFrame(rows)
    if frame.empty or "family" not in frame.columns:
        return {}
    out: dict[str, float] = {}
    for family, grp in frame.groupby("family", dropna=False):
        activation = pd.to_numeric(grp.get("activation_rate", 0.0), errors="coerce").fillna(0.0)
        exp = pd.to_numeric(grp.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
        dead_hits = int(((activation <= 0.0) & (exp <= 0.0)).sum())
        total = int(max(1, len(grp)))
        dead_ratio = float(dead_hits / total)
        weight = float(max(0.05, 1.0 - (0.8 * dead_ratio)))
        if float(activation.mean()) > 0.0:
            weight = float(min(1.25, weight + 0.10))
        out[str(family)] = weight
    return out


def prune_features_by_contribution(
    contribution_rows: list[dict[str, Any]],
    *,
    min_mean_gain: float = 0.0,
    keep_top: int = 64,
) -> dict[str, Any]:
    """Prune consistently useless features while preserving deterministic schema."""

    frame = pd.DataFrame(contribution_rows)
    if frame.empty:
        return {"kept_features": [], "dropped_features": [], "contribution_summary": {}}
    frame["feature"] = frame.get("feature", "").astype(str)
    frame["gain"] = pd.to_numeric(frame.get("gain", 0.0), errors="coerce").fillna(0.0)
    grouped = frame.groupby("feature", dropna=False)["gain"].agg(["mean", "count"]).reset_index()
    grouped = grouped.sort_values(["mean", "count", "feature"], ascending=[False, False, True])
    keep = grouped.loc[grouped["mean"] >= float(min_mean_gain), "feature"].astype(str).tolist()
    keep = keep[: int(max(1, keep_top))]
    if not keep:
        keep = grouped.head(int(max(1, min(keep_top, len(grouped)))))["feature"].astype(str).tolist()
    dropped = [str(item) for item in grouped["feature"].astype(str).tolist() if str(item) not in set(keep)]
    summary = {
        str(row["feature"]): {"mean_gain": float(row["mean"]), "count": int(row["count"])}
        for row in grouped.to_dict(orient="records")
    }
    return {
        "kept_features": keep,
        "dropped_features": dropped,
        "contribution_summary": summary,
    }


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["run_id"] = str(out.get("run_id", ""))
    out["generation"] = int(out.get("generation", 0))
    out["family"] = str(out.get("family", ""))
    out["feature_subset_signature"] = str(out.get("feature_subset_signature", ""))
    out["threshold_configuration"] = dict(out.get("threshold_configuration", {}) or {})
    out["raw_signal_count"] = int(out.get("raw_signal_count", 0))
    out["activation_rate"] = float(out.get("activation_rate", 0.0))
    out["top_reject_reason"] = str(out.get("top_reject_reason", "unknown"))
    out["cost_gate_fail_rate"] = float(out.get("cost_gate_fail_rate", 0.0))
    out["feasibility_fail_rate"] = float(out.get("feasibility_fail_rate", 0.0))
    out["final_trade_count"] = int(out.get("final_trade_count", 0))
    out["exp_lcb"] = float(out.get("exp_lcb", 0.0))
    out["stability_score"] = float(out.get("stability_score", 0.0))
    out["status"] = str(out.get("status", "active"))
    out["elite"] = bool(out.get("elite", False))
    motifs = out.get("failure_motif_tags", [])
    out["failure_motif_tags"] = [str(v) for v in motifs] if isinstance(motifs, list) else []
    return out
