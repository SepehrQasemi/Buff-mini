"""Stage-33 policy builder v3 (contextual + MTF-ready)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


@dataclass(frozen=True)
class PolicyV3Config:
    top_k_per_context: int = 3
    min_exp_lcb: float = 0.0
    min_usable_windows: int = 2
    w_min: float = 0.05
    w_max: float = 0.80
    conflict_mode: str = "net"


def build_policy_v3(
    finalists: pd.DataFrame,
    *,
    data_snapshot_id: str,
    data_snapshot_hash: str,
    config_hash: str,
    cfg: PolicyV3Config | None = None,
) -> dict[str, Any]:
    conf = cfg or PolicyV3Config()
    frame = finalists.copy() if isinstance(finalists, pd.DataFrame) else pd.DataFrame()
    contexts: dict[str, Any] = {}
    warnings: list[str] = []
    if frame.empty:
        warnings.append("no_finalists")
    else:
        for col, default in (
            ("candidate_id", ""),
            ("context", "GLOBAL"),
            ("timeframe", "1h"),
            ("htf", ""),
            ("ltf", ""),
            ("exp_lcb", 0.0),
            ("usable_windows", 0),
            ("symbol", ""),
        ):
            if col not in frame.columns:
                frame[col] = default
        frame["exp_lcb"] = pd.to_numeric(frame["exp_lcb"], errors="coerce").fillna(0.0)
        frame["usable_windows"] = pd.to_numeric(frame["usable_windows"], errors="coerce").fillna(0.0)
        frame["confidence"] = (frame["exp_lcb"].clip(lower=0.0) * np.log1p(frame["usable_windows"].clip(lower=0.0))).fillna(0.0)

        passed = frame.loc[
            (frame["exp_lcb"] >= float(conf.min_exp_lcb)) & (frame["usable_windows"] >= float(conf.min_usable_windows))
        ].copy()
        if passed.empty:
            warnings.append("no_rows_passed_policy_thresholds")
        for context, grp in passed.groupby("context", dropna=False):
            ranked = grp.sort_values(["confidence", "exp_lcb", "candidate_id"], ascending=[False, False, True]).head(int(max(1, conf.top_k_per_context)))
            weights = _bounded_weights(
                ranked["confidence"].to_numpy(dtype=float),
                w_min=float(conf.w_min),
                w_max=float(conf.w_max),
            )
            entries = []
            for idx, rec in enumerate(ranked.to_dict(orient="records")):
                entries.append(
                    {
                        "candidate_id": str(rec.get("candidate_id", "")),
                        "symbol": str(rec.get("symbol", "")),
                        "timeframe": str(rec.get("timeframe", "")),
                        "htf_bias_timeframe": str(rec.get("htf", rec.get("timeframe", ""))),
                        "ltf_trigger_timeframe": str(rec.get("ltf", rec.get("timeframe", ""))),
                        "weight": float(weights[idx]),
                        "exp_lcb": float(rec.get("exp_lcb", 0.0)),
                        "usable_windows": int(rec.get("usable_windows", 0)),
                    }
                )
            contexts[str(context)] = {
                "status": "OK",
                "candidates": entries,
                "weights": {str(item["candidate_id"]): float(item["weight"]) for item in entries},
            }

    payload: dict[str, Any] = {
        "version": "stage33_policy_v3",
        "generated_at": utc_now_compact(),
        "data_snapshot_id": str(data_snapshot_id),
        "data_snapshot_hash": str(data_snapshot_hash),
        "config_hash": str(config_hash),
        "conflict_mode": str(conf.conflict_mode),
        "contexts": contexts,
        "warnings": warnings,
        "mtf": {
            "supports_htf_bias": True,
            "supports_ltf_trigger": True,
        },
    }
    payload["policy_id"] = f"stage33_{stable_hash(payload, length=16)}"
    return payload


def render_policy_v3_spec_md(policy: dict[str, Any]) -> str:
    lines = [
        "# Stage-33 Policy v3 Spec",
        "",
        f"- policy_id: `{policy.get('policy_id', '')}`",
        f"- version: `{policy.get('version', '')}`",
        f"- generated_at: `{policy.get('generated_at', '')}`",
        f"- data_snapshot_id: `{policy.get('data_snapshot_id', '')}`",
        f"- data_snapshot_hash: `{policy.get('data_snapshot_hash', '')}`",
        f"- config_hash: `{policy.get('config_hash', '')}`",
        f"- conflict_mode: `{policy.get('conflict_mode', '')}`",
        "",
        "## Context Rules",
    ]
    contexts = dict(policy.get("contexts", {}))
    if not contexts:
        lines.append("- No active context mappings.")
    else:
        for context in sorted(contexts.keys()):
            row = dict(contexts.get(context, {}))
            lines.append(f"### {context}")
            lines.append(f"- status: `{row.get('status', '')}`")
            candidates = list(row.get("candidates", []))
            if not candidates:
                lines.append("- candidates: none")
                continue
            lines.append("| candidate_id | symbol | timeframe | HTF | LTF | weight | exp_lcb | usable_windows |")
            lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: |")
            for item in candidates:
                lines.append(
                    f"| {item.get('candidate_id','')} | {item.get('symbol','')} | {item.get('timeframe','')} | "
                    f"{item.get('htf_bias_timeframe','')} | {item.get('ltf_trigger_timeframe','')} | "
                    f"{float(item.get('weight', 0.0)):.6f} | {float(item.get('exp_lcb', 0.0)):.6f} | {int(item.get('usable_windows', 0))} |"
                )
    return "\n".join(lines).strip() + "\n"


def write_policy_v3(policy: dict[str, Any], *, out_dir: Path) -> tuple[Path, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "policy.json"
    md_path = target / "policy_spec.md"
    json_path.write_text(json.dumps(policy, indent=2, allow_nan=False), encoding="utf-8")
    md_path.write_text(render_policy_v3_spec_md(policy), encoding="utf-8")
    return json_path, md_path


def _bounded_weights(values: np.ndarray, *, w_min: float, w_max: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    if np.all(arr <= 0.0):
        arr = np.ones_like(arr)
    arr = arr / np.sum(arr)
    arr = np.clip(arr, float(w_min), float(w_max))
    arr = arr / np.sum(arr)
    return arr.astype(float)

