"""Stage-13.5 family composer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def normalize_weights(weights: dict[str, float], enabled_families: list[str]) -> dict[str, float]:
    out = {str(name): max(0.0, float(weights.get(str(name), 0.0))) for name in enabled_families}
    total = float(sum(out.values()))
    if total <= 0:
        uniform = 1.0 / float(max(1, len(enabled_families)))
        return {str(name): uniform for name in enabled_families}
    return {name: value / total for name, value in out.items()}


def compose_signals(
    *,
    family_outputs: dict[str, pd.DataFrame],
    mode: str,
    weights: dict[str, float] | None = None,
    gated_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compose per-family outputs into a single standardized signal frame."""

    if not family_outputs:
        raise ValueError("family_outputs must be non-empty")
    names = sorted(family_outputs.keys())
    ref = family_outputs[names[0]].copy()
    idx = ref.index
    aligned = {name: family_outputs[name].reindex(idx).copy() for name in names}
    mode_text = str(mode).strip().lower()

    if mode_text == "vote":
        vote = np.zeros(len(idx), dtype=float)
        conf = np.zeros(len(idx), dtype=float)
        for name in names:
            direction = pd.to_numeric(aligned[name].get("direction", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            confidence = pd.to_numeric(aligned[name].get("confidence", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            vote += direction * np.clip(confidence, 0.0, 1.0)
            conf += np.clip(confidence, 0.0, 1.0)
        score = vote / np.maximum(conf, 1e-12)
    elif mode_text == "weighted_sum":
        norm = normalize_weights(weights or {}, names)
        score = np.zeros(len(idx), dtype=float)
        for name in names:
            score += float(norm[name]) * pd.to_numeric(aligned[name].get("score", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    elif mode_text == "gated":
        cfg = dict(gated_config or {})
        gate_family = str(cfg.get("gate_family", "volatility"))
        gate_threshold = float(cfg.get("gate_threshold", 0.2))
        base_weights = normalize_weights(weights or {}, names)
        base = np.zeros(len(idx), dtype=float)
        for name in names:
            if name == gate_family:
                continue
            base += float(base_weights.get(name, 0.0)) * pd.to_numeric(aligned[name].get("score", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        gate_score = pd.to_numeric(aligned.get(gate_family, ref).get("score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        score = np.where(np.abs(gate_score) >= gate_threshold, base, 0.5 * base)
    else:
        raise ValueError("mode must be one of: vote, weighted_sum, gated")

    score_series = pd.Series(np.clip(score, -1.0, 1.0), index=idx, dtype=float)
    threshold = float(np.clip(float((gated_config or {}).get("entry_threshold", 0.25)), 0.0, 1.0))
    direction = np.where(score_series >= threshold, 1, np.where(score_series <= -threshold, -1, 0))
    signal = pd.Series(direction, index=idx, dtype=int).shift(1).fillna(0).astype(int)
    return pd.DataFrame(
        {
            "score": score_series,
            "direction": pd.Series(direction, index=idx, dtype=int),
            "confidence": score_series.abs().clip(0.0, 1.0),
            "reasons": pd.Series([f"composer:{mode_text}"] * len(idx), index=idx, dtype="object"),
            "long_entry": pd.Series(direction, index=idx, dtype=int) == 1,
            "short_entry": pd.Series(direction, index=idx, dtype=int) == -1,
            "signal": signal,
            "signal_family": f"composer:{mode_text}",
        }
    )

