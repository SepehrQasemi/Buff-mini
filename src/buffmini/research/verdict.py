"""Stage-103 final edge-existence verdict aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from buffmini.utils.hashing import stable_hash


def load_stage_payloads(docs_dir: Path, *, stages: list[int]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for stage in stages:
        path = docs_dir / f"stage{stage}_summary.json"
        if path.exists():
            payloads[str(stage)] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def derive_final_edge_verdict(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stage95 = dict(payloads.get("95", {}))
    stage96 = dict(payloads.get("96", {}))
    stage100 = dict(payloads.get("100", {}))
    stage101 = dict(payloads.get("101", {}))
    stage102 = dict(payloads.get("102", {}))

    truth_counts = dict(stage100.get("truth_counts", {}))
    rescue_counts = dict(stage102.get("classification_counts", {}))
    robust_rows = int(truth_counts.get("robust_signal_exists", 0))
    replay_fragile_rows = int(truth_counts.get("replay_fragile_signal_only", 0))
    data_block_rows = int(truth_counts.get("data_blocks_interpretation", 0))
    beats_all_controls = int(stage101.get("candidate_beats_all_controls_count", 0))
    beats_majority_controls = int(stage101.get("candidate_beats_majority_controls_count", 0))
    stage95_dead_weight = int(len(stage95.get("dead_weight_families", [])))
    stage96_fitness_rows = list(((stage96.get("fitness_after") or {}).get("rows") or []))
    if stage96_fitness_rows:
        stage96_canonical_rows = int(sum(1 for row in stage96_fitness_rows if bool(row.get("canonical_usable", False))))
    else:
        stage96_canonical_rows = int(sum(1 for row in stage96.get("repair_rows", []) if bool(row.get("strict_usable_after", False))))

    if robust_rows > 0 or int(rescue_counts.get("rescueable", 0)) > 0:
        verdict = "ROBUST_CANDIDATE_FOUND"
    elif data_block_rows > 0 and replay_fragile_rows == 0:
        verdict = "DATA_OR_SCOPE_BLOCKS_STRONGER_CONCLUSION"
    elif replay_fragile_rows > 0:
        if beats_majority_controls > 0 or int(rescue_counts.get("transfer_blocked", 0)) > 0:
            verdict = "PROMISING_BUT_UNPROVEN_CANDIDATES_FOUND"
        elif stage95_dead_weight > 0 and beats_all_controls == 0:
            verdict = "GENERATOR_OR_SEARCH_FORMALISM_STILL_INSUFFICIENT"
        else:
            verdict = "WEAK_REGIME_LOCAL_MECHANISMS_FOUND"
    elif stage96_canonical_rows <= 0 or data_block_rows > 0:
        verdict = "DATA_OR_SCOPE_BLOCKS_STRONGER_CONCLUSION"
    else:
        verdict = "NO_INTERPRETABLE_EDGE_FOUND_IN_SCOPE"

    evidence_table = [
        {"stage": "95", "metric": "dead_weight_family_count", "value": stage95_dead_weight},
        {"stage": "96", "metric": "canonical_usable_rows", "value": stage96_canonical_rows},
        {"stage": "100", "metric": "replay_fragile_signal_rows", "value": replay_fragile_rows},
        {"stage": "100", "metric": "data_block_rows", "value": data_block_rows},
        {"stage": "101", "metric": "candidates_beating_all_controls", "value": beats_all_controls},
        {"stage": "101", "metric": "candidates_beating_majority_controls", "value": beats_majority_controls},
        {"stage": "102", "metric": "rescueable_count", "value": int(rescue_counts.get("rescueable", 0))},
        {"stage": "102", "metric": "transfer_blocked_count", "value": int(rescue_counts.get("transfer_blocked", 0))},
        {"stage": "102", "metric": "still_generator_weak_count", "value": int(rescue_counts.get("still_generator_weak", 0))},
    ]
    summary = {
        "final_edge_verdict": verdict,
        "evidence_table": evidence_table,
        "supporting_counts": {
            "truth_counts": truth_counts,
            "rescue_counts": rescue_counts,
            "beats_all_controls": beats_all_controls,
            "beats_majority_controls": beats_majority_controls,
        },
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    return summary
