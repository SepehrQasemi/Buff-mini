"""Stage-28 policy composer v2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.ui.components.library import LIBRARY_DIR, load_library_index, save_library_index
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact

_BUILD_TS_UTC = utc_now_compact()


@dataclass(frozen=True)
class PolicyV2Config:
    top_k_per_context: int = 3
    min_occurrences_context: int = 50
    min_trades_context: int = 30
    min_exp_lcb: float = 0.0
    w_min: float = 0.05
    w_max: float = 0.80
    conflict_mode: str = "net"


def build_policy_v2(
    finalists: pd.DataFrame,
    *,
    data_snapshot_id: str,
    data_snapshot_hash: str,
    config_hash: str,
    cfg: PolicyV2Config | None = None,
) -> dict[str, Any]:
    """Build deployable context->candidate policy from Stage-C finalists."""

    conf = cfg or PolicyV2Config()
    work = finalists.copy() if isinstance(finalists, pd.DataFrame) else pd.DataFrame()
    if work.empty:
        policy_id = stable_hash(
            {
                "snapshot": data_snapshot_id,
                "snapshot_hash": data_snapshot_hash,
                "cfg": config_hash,
                "empty": True,
            },
            length=16,
        )
        return {
            "policy_id": f"stage28_{policy_id}",
            "version": "stage28_policy_v2",
            "generated_at": _BUILD_TS_UTC,
            "data_snapshot_id": str(data_snapshot_id),
            "data_snapshot_hash": str(data_snapshot_hash),
            "config_hash": str(config_hash),
            "conflict_mode": str(conf.conflict_mode),
            "contexts": {},
            "warnings": ["no_finalists"],
        }

    work = _normalize_finalists(work)
    work = work.loc[
        (work["context_occurrences"] >= float(conf.min_occurrences_context))
        & (work["trades_in_context"] >= float(conf.min_trades_context))
        & (work["exp_lcb"] > float(conf.min_exp_lcb))
    ].copy()

    warnings: list[str] = []
    contexts: dict[str, Any] = {}
    if work.empty:
        warnings.append("no_rows_passed_policy_thresholds")
    else:
        for ctx, grp in work.groupby("context", dropna=False):
            context = str(ctx)
            ranked = grp.sort_values(
                ["confidence_score", "exp_lcb", "trades_in_context", "candidate_id"],
                ascending=[False, False, False, True],
            ).head(int(max(1, conf.top_k_per_context)))
            if ranked.empty:
                warnings.append(f"{context}:empty_after_ranking")
                contexts[context] = {"status": "EMPTY", "candidates": [], "weights": {}}
                continue
            weights = _bounded_weights(
                ranked["confidence_score"].to_numpy(dtype=float),
                w_min=float(conf.w_min),
                w_max=float(conf.w_max),
            )
            candidates = []
            weight_map: dict[str, float] = {}
            for idx, row in enumerate(ranked.to_dict(orient="records")):
                cand_id = str(row.get("candidate_id", ""))
                weight = float(weights[idx])
                weight_map[cand_id] = weight
                candidates.append(
                    {
                        "candidate_id": cand_id,
                        "candidate": str(row.get("candidate", "")),
                        "symbol": str(row.get("symbol", "")),
                        "timeframe": str(row.get("timeframe", "")),
                        "family": str(row.get("family", "")),
                        "weight": weight,
                        "confidence_score": float(row.get("confidence_score", 0.0)),
                        "exp_lcb": float(row.get("exp_lcb", 0.0)),
                        "trades_in_context": int(row.get("trades_in_context", 0)),
                        "context_occurrences": int(row.get("context_occurrences", 0)),
                    }
                )
            contexts[context] = {
                "status": "OK",
                "weights": weight_map,
                "candidates": candidates,
            }

    policy_payload = {
        "version": "stage28_policy_v2",
        "generated_at": _BUILD_TS_UTC,
        "data_snapshot_id": str(data_snapshot_id),
        "data_snapshot_hash": str(data_snapshot_hash),
        "config_hash": str(config_hash),
        "conflict_mode": str(conf.conflict_mode).strip().lower(),
        "contexts": contexts,
        "warnings": warnings,
    }
    policy_payload["policy_id"] = f"stage28_{stable_hash(policy_payload, length=16)}"
    return policy_payload


def compose_policy_signal_v2(
    *,
    frame: pd.DataFrame,
    policy: dict[str, Any],
    candidate_signals: dict[str, pd.Series],
) -> tuple[pd.Series, pd.DataFrame]:
    """Compose final policy signal from context-specific weighted candidates."""

    work = frame.copy()
    state = work.get("ctx_state", pd.Series("RANGE", index=work.index)).astype(str)
    mode = str(policy.get("conflict_mode", "net")).strip().lower()
    contexts = dict(policy.get("contexts", {}))
    out = np.zeros(len(work), dtype=int)
    rows: list[dict[str, Any]] = []

    for i in range(len(work)):
        ctx = str(state.iloc[i])
        ctx_policy = dict(contexts.get(ctx, {}))
        candidates = list(ctx_policy.get("candidates", []))
        long_score = 0.0
        short_score = 0.0
        net_score = 0.0
        active: list[str] = []
        for item in candidates:
            cand_id = str(item.get("candidate_id", ""))
            if not cand_id:
                continue
            series = pd.to_numeric(candidate_signals.get(cand_id, pd.Series(0, index=work.index)), errors="coerce").fillna(0)
            if i >= len(series):
                continue
            val = float(series.iloc[i])
            weight = float(item.get("weight", 0.0))
            net_score += weight * val
            if val > 0:
                long_score += weight * val
            elif val < 0:
                short_score += weight * abs(val)
            if abs(val) > 0:
                active.append(cand_id)

        if mode == "hedge":
            signal = 1 if long_score >= short_score and long_score > 0 else -1 if short_score > long_score else 0
        elif mode == "isolated":
            signal = 1 if net_score > 0 else -1 if net_score < 0 else 0
        else:
            signal = 1 if net_score > 0 else -1 if net_score < 0 else 0
        out[i] = int(signal)
        rows.append(
            {
                "timestamp": _timestamp_at(work, i),
                "context": ctx,
                "net_score": float(net_score),
                "long_score": float(long_score),
                "short_score": float(short_score),
                "active_candidates": ",".join(sorted(active)),
                "final_signal": int(signal),
                "conflict_mode": mode,
            }
        )

    signal_series = pd.Series(out, index=work.index, dtype=int).shift(1).fillna(0).astype(int)
    return signal_series, pd.DataFrame(rows)


def export_policy_to_library(
    policy: dict[str, Any],
    *,
    library_dir: Path = LIBRARY_DIR,
) -> dict[str, Any]:
    """Persist stage-28 policy into local library index."""

    base = Path(library_dir)
    policy_id = str(policy.get("policy_id", ""))
    if not policy_id:
        raise ValueError("policy_id missing")
    policy_dir = base / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(policy)
    out_path = policy_dir / f"{policy_id}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    card = {
        "policy_id": policy_id,
        "version": str(policy.get("version", "")),
        "generated_at": str(policy.get("generated_at", "")),
        "data_snapshot_id": str(policy.get("data_snapshot_id", "")),
        "data_snapshot_hash": str(policy.get("data_snapshot_hash", "")),
        "config_hash": str(policy.get("config_hash", "")),
        "conflict_mode": str(policy.get("conflict_mode", "net")),
        "contexts_count": int(len(dict(policy.get("contexts", {})))),
    }
    index_payload = load_library_index(base)
    policies = list(index_payload.get("policies", []))
    policies = [item for item in policies if str(item.get("policy_id", "")) != policy_id]
    policies.append(card)
    index_payload["policies"] = sorted(policies, key=lambda item: str(item.get("policy_id", "")))
    save_library_index(index_payload, base)
    return {"policy_card": card, "path": str(out_path.as_posix())}


def render_policy_spec_md(policy: dict[str, Any]) -> str:
    """Render human-readable policy specification markdown."""

    lines = [
        "# Stage-28 Policy Spec",
        "",
        f"- policy_id: `{policy.get('policy_id', '')}`",
        f"- version: `{policy.get('version', '')}`",
        f"- generated_at: `{policy.get('generated_at', '')}`",
        f"- data_snapshot_id: `{policy.get('data_snapshot_id', '')}`",
        f"- data_snapshot_hash: `{policy.get('data_snapshot_hash', '')}`",
        f"- config_hash: `{policy.get('config_hash', '')}`",
        f"- conflict_mode: `{policy.get('conflict_mode', '')}`",
        "",
        "## Context Map",
    ]
    contexts = dict(policy.get("contexts", {}))
    if not contexts:
        lines.append("- No contexts selected.")
    else:
        for context in sorted(contexts.keys()):
            ctx = dict(contexts.get(context, {}))
            lines.append(f"### {context}")
            lines.append(f"- status: `{ctx.get('status', '')}`")
            candidates = list(ctx.get("candidates", []))
            if not candidates:
                lines.append("- candidates: none")
                continue
            lines.append("| candidate_id | candidate | symbol | timeframe | weight | exp_lcb | trades |")
            lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
            for row in candidates:
                lines.append(
                    "| {candidate_id} | {candidate} | {symbol} | {timeframe} | {weight:.6f} | {exp_lcb:.6f} | {trades_in_context} |".format(
                        candidate_id=str(row.get("candidate_id", "")),
                        candidate=str(row.get("candidate", "")),
                        symbol=str(row.get("symbol", "")),
                        timeframe=str(row.get("timeframe", "")),
                        weight=float(row.get("weight", 0.0)),
                        exp_lcb=float(row.get("exp_lcb", 0.0)),
                        trades_in_context=int(row.get("trades_in_context", 0)),
                    )
                )
    warnings = list(policy.get("warnings", []))
    lines.append("")
    lines.append("## Warnings")
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- none")
    return "\n".join(lines).strip() + "\n"


def _normalize_finalists(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col, default in (
        ("candidate_id", ""),
        ("candidate", ""),
        ("symbol", ""),
        ("timeframe", ""),
        ("family", "unknown"),
        ("context", "UNKNOWN"),
    ):
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].astype(str)
    for col in ("exp_lcb", "expectancy", "cost_sensitivity", "trades_in_context", "context_occurrences"):
        out[col] = pd.to_numeric(out.get(col, 0.0), errors="coerce").fillna(0.0)
    if "confidence_score" not in out.columns:
        out["confidence_score"] = _confidence_score(out)
    else:
        out["confidence_score"] = pd.to_numeric(out["confidence_score"], errors="coerce").fillna(0.0)
    generated_ids = [
        stable_hash(
            {
                "candidate": row.get("candidate", ""),
                "symbol": row.get("symbol", ""),
                "timeframe": row.get("timeframe", ""),
                "context": row.get("context", ""),
            },
            length=12,
        )
        for row in out.to_dict(orient="records")
    ]
    cid = out["candidate_id"].astype(str).str.strip()
    out["candidate_id"] = [existing if existing else generated for existing, generated in zip(cid.tolist(), generated_ids, strict=False)]
    return out


def _confidence_score(frame: pd.DataFrame) -> pd.Series:
    exp_lcb = np.maximum(pd.to_numeric(frame.get("exp_lcb", 0.0), errors="coerce").fillna(0.0), 0.0)
    trades = pd.to_numeric(frame.get("trades_in_context", 0.0), errors="coerce").fillna(0.0)
    occurrences = pd.to_numeric(frame.get("context_occurrences", 0.0), errors="coerce").fillna(0.0)
    sensitivity = pd.to_numeric(frame.get("cost_sensitivity", 0.0), errors="coerce").fillna(0.0)
    evidence = np.log1p(np.maximum(trades, 0.0)) + 0.2 * np.log1p(np.maximum(occurrences, 0.0))
    stability = np.clip(1.0 - np.abs(sensitivity), 0.0, 1.0)
    return exp_lcb * (0.5 + 0.5 * stability) + 0.05 * evidence


def _bounded_weights(confidence: np.ndarray, *, w_min: float, w_max: float) -> np.ndarray:
    raw = np.asarray(confidence, dtype=float)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    raw = np.maximum(raw, 0.0)
    if raw.size == 0:
        return raw
    if float(raw.sum()) <= 0.0:
        raw = np.ones_like(raw)
    weights = raw / float(raw.sum())
    clipped = np.clip(weights, float(w_min), float(w_max))
    total = float(clipped.sum())
    if total <= 0.0:
        clipped = np.ones_like(clipped) / float(len(clipped))
    else:
        clipped = clipped / total
    return clipped


def _timestamp_at(frame: pd.DataFrame, idx: int) -> str:
    if "timestamp" not in frame.columns:
        return ""
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if idx >= len(ts):
        return ""
    value = ts.iloc[idx]
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).tz_convert("UTC").isoformat()
