"""Stage-15 alpha-v2 orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buffmini.alpha_v2.contracts import AlphaRole, SignalContract, validate_contract_output


@dataclass(frozen=True)
class OrchestratorConfig:
    """Deterministic orchestrator settings."""

    entry_threshold: float = 0.25
    min_confidence: float = 0.05
    riskgate_hard: bool = False
    riskgate_soft_floor: float = 0.25


def run_orchestrator(
    *,
    frame: pd.DataFrame,
    contracts: list[SignalContract],
    seed: int,
    config: dict[str, Any],
    orchestrator_cfg: OrchestratorConfig | None = None,
    context_weights: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Route role-scores and build final order intents."""

    cfg = orchestrator_cfg or OrchestratorConfig()
    if frame.empty:
        empty = pd.DataFrame(
            columns=["timestamp", "side", "score", "confidence", "activation_multiplier", "reasons"]
        )
        stats = {
            "contracts_evaluated": 0,
            "nonzero_entry_scores": 0,
            "mean_activation_multiplier": 1.0,
            "pct_not_neutral_multiplier": 0.0,
        }
        return empty, stats

    by_role: dict[AlphaRole, list[pd.Series]] = {
        AlphaRole.ENTRY: [],
        AlphaRole.CONFIRM: [],
        AlphaRole.RISK_GATE: [],
        AlphaRole.EXIT_MODIFIER: [],
        AlphaRole.SIZING_MODIFIER: [],
    }
    reasons_parts: list[np.ndarray] = []

    for component in contracts:
        score = component.compute_score(frame, seed=int(seed), config=config)
        validate_contract_output(score)
        clipped = SignalContract.clip_scores(score)
        by_role[component.role].append(clipped)
        # Short reason tag only when component is active on bar.
        tag = np.where(clipped.abs().to_numpy(dtype=float) > 1e-12, component.name, "")
        reasons_parts.append(tag)

    entry_score = _avg_score(by_role[AlphaRole.ENTRY], len(frame))
    confirm_score = _avg_score(by_role[AlphaRole.CONFIRM], len(frame))
    risk_score = _avg_score(by_role[AlphaRole.RISK_GATE], len(frame))
    sizing_score = _avg_score(by_role[AlphaRole.SIZING_MODIFIER], len(frame))

    confirm_mod = np.clip(0.5 + 0.5 * np.maximum(confirm_score, 0.0), 0.5, 1.0)
    if cfg.riskgate_hard:
        risk_mod = np.where(risk_score < 0.0, 0.0, 1.0)
    else:
        risk_mod = np.clip(1.0 + risk_score, cfg.riskgate_soft_floor, 1.0)
    sizing_mod = np.clip(1.0 + 0.25 * sizing_score, 0.5, 1.5)

    final_score = entry_score * confirm_mod * risk_mod
    if context_weights is not None:
        c = pd.to_numeric(context_weights, errors="coerce").fillna(1.0).to_numpy(dtype=float)
        final_score = final_score * np.clip(c, 0.25, 2.0)
    activation_multiplier = np.clip(confirm_mod * risk_mod * sizing_mod, 0.0, 2.0)
    confidence = np.clip(np.abs(final_score) * activation_multiplier, 0.0, 1.0)

    side = np.where(final_score >= cfg.entry_threshold, 1, np.where(final_score <= -cfg.entry_threshold, -1, 0))
    side = np.where(confidence >= cfg.min_confidence, side, 0).astype(int)

    reasons = _join_reasons(reasons_parts)
    intents = pd.DataFrame(
        {
            "timestamp": frame["timestamp"].copy(),
            "side": pd.Series(side, index=frame.index, dtype=int),
            "score": pd.Series(final_score, index=frame.index, dtype=float),
            "confidence": pd.Series(confidence, index=frame.index, dtype=float),
            "activation_multiplier": pd.Series(activation_multiplier, index=frame.index, dtype=float),
            "reasons": pd.Series(reasons, index=frame.index, dtype="object"),
        }
    )

    stats = {
        "contracts_evaluated": int(len(contracts)),
        "nonzero_entry_scores": int(np.count_nonzero(np.abs(entry_score) > 1e-12)),
        "mean_activation_multiplier": float(np.mean(activation_multiplier)),
        "pct_not_neutral_multiplier": float(np.mean(np.abs(activation_multiplier - 1.0) > 1e-12) * 100.0),
    }
    return intents, stats


def _avg_score(parts: list[pd.Series], n_rows: int) -> np.ndarray:
    if not parts:
        return np.zeros(n_rows, dtype=float)
    stacked = np.column_stack([pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float) for s in parts])
    return np.mean(stacked, axis=1, dtype=float)


def _join_reasons(reasons_parts: list[np.ndarray]) -> np.ndarray:
    if not reasons_parts:
        return np.full(0, "", dtype=object)
    n = len(reasons_parts[0])
    joined = []
    for i in range(n):
        tags = [str(part[i]) for part in reasons_parts if str(part[i])]
        joined.append("|".join(tags[:4]))
    return np.asarray(joined, dtype=object)

