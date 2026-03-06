"""Stage-39 layered upstream candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from buffmini.utils.hashing import stable_hash


@dataclass(frozen=True)
class LayeredCandidateOutput:
    """Layer-A/B/C candidate frames."""

    layer_a: pd.DataFrame
    layer_b: pd.DataFrame
    layer_c: pd.DataFrame


GRAMMAR_BRANCHES: tuple[dict[str, Any], ...] = (
    {
        "branch": "funding_divergence_continuation",
        "families": ("funding", "flow"),
        "contexts": ("trend", "flow-dominant", "funding-stress"),
        "trigger": "funding_price_divergence_up",
        "confirmation": "directional_continuation",
    },
    {
        "branch": "funding_extreme_reversal_attempt",
        "families": ("funding", "price"),
        "contexts": ("funding-stress", "exhaustion", "sentiment-extreme"),
        "trigger": "funding_extreme",
        "confirmation": "mean_reversion_confirmation",
    },
    {
        "branch": "taker_imbalance_squeeze_break",
        "families": ("taker_buy_sell", "flow", "volatility"),
        "contexts": ("squeeze", "flow-dominant", "shock"),
        "trigger": "taker_imbalance_burst",
        "confirmation": "squeeze_break_direction",
    },
    {
        "branch": "long_short_extreme_exhaustion",
        "families": ("long_short_ratio", "price"),
        "contexts": ("sentiment-extreme", "exhaustion", "range"),
        "trigger": "ls_ratio_extreme",
        "confirmation": "exhaustion_reversal",
    },
    {
        "branch": "vol_compression_flow_burst",
        "families": ("volatility", "flow"),
        "contexts": ("squeeze", "shock", "flow-dominant"),
        "trigger": "volatility_compression",
        "confirmation": "flow_burst",
    },
)


CONTEXT_MAP: dict[str, str] = {
    "TREND": "trend",
    "RANGE": "range",
    "VOL_EXPANSION": "squeeze",
    "VOLUME_SHOCK": "shock",
    "EXHAUSTION": "exhaustion",
    "FLOW_DOMINANT": "flow-dominant",
    "FUNDING_STRESS": "funding-stress",
    "SENTIMENT_EXTREME": "sentiment-extreme",
}


def widen_context_label(label: str) -> str:
    """Map narrow context labels to Stage-39 broad contexts."""

    key = str(label).strip().upper()
    if key in CONTEXT_MAP:
        return str(CONTEXT_MAP[key])
    if "TREND" in key:
        return "trend"
    if "RANGE" in key:
        return "range"
    if "VOL" in key and "EXP" in key:
        return "squeeze"
    if "SHOCK" in key:
        return "shock"
    return "flow-dominant"


def build_layered_candidates(finalists: pd.DataFrame, *, seed: int = 42) -> LayeredCandidateOutput:
    """Build Layer A/B/C candidates deterministically from Stage-28 finalists."""

    base = _normalize_finalists(finalists)
    if base.empty:
        base = pd.DataFrame(
            [
                {"candidate": "fallback_flow", "family": "flow", "context": "RANGE"},
                {"candidate": "fallback_vol", "family": "volatility", "context": "VOL_EXPANSION"},
            ]
        )
        base = _normalize_finalists(base)

    context_counts = base["broad_context"].value_counts(dropna=False).to_dict()
    family_counts = base["family"].value_counts(dropna=False).to_dict()
    total = max(1, int(len(base)))

    rows: list[dict[str, Any]] = []
    for base_row in base.to_dict(orient="records"):
        base_context = str(base_row.get("broad_context", "flow-dominant"))
        base_family = str(base_row.get("family", "flow"))
        for grammar in GRAMMAR_BRANCHES:
            allowed_contexts = tuple(str(v) for v in grammar["contexts"])
            if base_context not in set(allowed_contexts):
                continue
            for family in tuple(str(v) for v in grammar["families"]):
                context_prior = float(context_counts.get(base_context, 0)) / float(total)
                family_prior = float(family_counts.get(family, 0)) / float(total)
                novelty = 0.2 if family not in set(family_counts.keys()) else 0.0
                branch_bonus = 0.05 * (1 + int(len(rows) % 3))
                layer_score = float(round(context_prior * 0.55 + family_prior * 0.35 + novelty + branch_bonus, 6))
                candidate_id = stable_hash(
                    {
                        "seed": int(seed),
                        "src": str(base_row.get("candidate_id", "")),
                        "branch": str(grammar["branch"]),
                        "family": family,
                        "context": base_context,
                        "idx": int(len(rows)),
                    },
                    length=14,
                )
                rows.append(
                    {
                        "candidate_id": f"s39_{candidate_id}",
                        "source_candidate_id": str(base_row.get("candidate_id", "")),
                        "source_candidate": str(base_row.get("candidate", "")),
                        "source_family": base_family,
                        "family": family,
                        "base_context": str(base_row.get("context", "")),
                        "broad_context": base_context,
                        "branch": str(grammar["branch"]),
                        "precondition": base_context,
                        "trigger": str(grammar["trigger"]),
                        "confirmation": str(grammar["confirmation"]),
                        "layer_score": layer_score,
                    }
                )

    layer_a = pd.DataFrame(rows).sort_values(["branch", "family", "candidate_id"], ascending=[True, True, True]).reset_index(drop=True)
    if layer_a.empty:
        layer_a = pd.DataFrame(columns=["candidate_id", "family", "broad_context", "branch", "layer_score"])

    # Layer B: light quality pruning (keep score >= median within branch + remove duplicate family/context combos).
    if layer_a.empty:
        layer_b = layer_a.copy()
    else:
        med = layer_a.groupby("branch", dropna=False)["layer_score"].transform("median")
        keep = layer_a.loc[layer_a["layer_score"] >= med, :].copy()
        keep = keep.sort_values(["layer_score", "candidate_id"], ascending=[False, True])
        keep = keep.drop_duplicates(subset=["branch", "family", "broad_context"], keep="first")
        layer_b = keep.reset_index(drop=True)

    # Layer C: policy-ready shortlist (top-N per family + top context diversity).
    if layer_b.empty:
        layer_c = layer_b.copy()
    else:
        ranked = layer_b.sort_values(["family", "layer_score", "candidate_id"], ascending=[True, False, True]).copy()
        ranked["family_rank"] = ranked.groupby("family", dropna=False).cumcount()
        shortlisted = ranked.loc[ranked["family_rank"] < 4, :].drop(columns=["family_rank"])
        shortlisted = shortlisted.sort_values(
            ["layer_score", "branch", "family", "candidate_id"],
            ascending=[False, True, True, True],
        ).head(96)
        layer_c = shortlisted.reset_index(drop=True)

    return LayeredCandidateOutput(layer_a=layer_a, layer_b=layer_b, layer_c=layer_c)


def summarize_layered_candidates(output: LayeredCandidateOutput) -> dict[str, Any]:
    """Summarize Stage-39 layered candidate flow."""

    layer_a = output.layer_a.copy()
    layer_b = output.layer_b.copy()
    layer_c = output.layer_c.copy()
    return {
        "raw_candidate_count": int(layer_a.shape[0]),
        "light_pruned_count": int(layer_b.shape[0]),
        "shortlisted_count": int(layer_c.shape[0]),
        "family_counts_layer_a": _counts(layer_a, key="family"),
        "family_counts_layer_b": _counts(layer_b, key="family"),
        "family_counts_layer_c": _counts(layer_c, key="family"),
        "context_counts_layer_a": _counts(layer_a, key="broad_context"),
        "context_counts_layer_b": _counts(layer_b, key="broad_context"),
        "context_counts_layer_c": _counts(layer_c, key="broad_context"),
        "nonzero_branches": sorted(
            [
                str(branch)
                for branch, count in _counts(layer_a, key="branch").items()
                if int(count) > 0
            ]
        ),
    }


def _normalize_finalists(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if out.empty:
        return pd.DataFrame(columns=["candidate_id", "candidate", "family", "context", "broad_context"])
    out["candidate_id"] = out.get("candidate_id", "").astype(str).fillna("")
    out["candidate"] = out.get("candidate", "").astype(str).fillna("")
    out["family"] = out.get("family", "flow").astype(str).fillna("flow")
    out["context"] = out.get("context", "RANGE").astype(str).fillna("RANGE")
    out["broad_context"] = out["context"].map(widen_context_label)
    return out


def _counts(frame: pd.DataFrame, *, key: str) -> dict[str, int]:
    if frame.empty or key not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[key].astype(str).value_counts(dropna=False).to_dict().items()}

