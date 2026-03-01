"""Stage-21 bounded search with pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class SearchConfigV2:
    max_evaluations: int = 5000
    beam_width: int = 64
    seed: int = 42


def bounded_search(
    *,
    candidate_space: list[dict[str, Any]],
    evaluate_fn: Callable[[dict[str, Any]], dict[str, Any]],
    cfg: SearchConfigV2,
) -> dict[str, Any]:
    """Deterministic bounded search with stage-A/B/C style pruning."""

    rng = np.random.default_rng(int(cfg.seed))
    order = np.arange(len(candidate_space), dtype=int)
    rng.shuffle(order)
    evaluated = 0
    pruned = 0
    kept: list[dict[str, Any]] = []
    prune_reasons: dict[str, int] = {}

    for idx in order:
        if evaluated >= int(cfg.max_evaluations):
            break
        candidate = dict(candidate_space[int(idx)])
        result = dict(evaluate_fn(candidate))
        evaluated += 1
        if not bool(result.get("valid", True)):
            pruned += 1
            reason = str(result.get("reason", "PRUNED"))
            prune_reasons[reason] = prune_reasons.get(reason, 0) + 1
            continue
        kept.append({"candidate": candidate, "result": result})
        kept.sort(key=lambda item: float(item["result"].get("score", -1e12)), reverse=True)
        if len(kept) > int(cfg.beam_width):
            kept = kept[: int(cfg.beam_width)]

    return {
        "evaluated_count": int(evaluated),
        "pruned_count": int(pruned),
        "top_candidates": kept,
        "prune_reasons": prune_reasons,
    }

