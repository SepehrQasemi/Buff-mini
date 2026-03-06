from __future__ import annotations

import pandas as pd

from buffmini.stage39.signal_generation import build_layered_candidates, summarize_layered_candidates


def _finalists_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"candidate_id": "c1", "candidate": "MomentumBurst", "family": "flow", "context": "VOLUME_SHOCK"},
            {"candidate_id": "c2", "candidate": "VolExpansionContinuation", "family": "volatility", "context": "VOL_EXPANSION"},
            {"candidate_id": "c3", "candidate": "FailedBreakReversal", "family": "price", "context": "RANGE"},
        ]
    )


def test_stage39_layered_candidate_lineage_is_monotonic() -> None:
    out = build_layered_candidates(_finalists_fixture(), seed=42)
    summary = summarize_layered_candidates(out)
    assert int(summary["raw_candidate_count"]) >= int(summary["light_pruned_count"]) >= int(summary["shortlisted_count"])
    assert int(summary["raw_candidate_count"]) > 0
    assert len(list(summary["nonzero_branches"])) > 0


def test_stage39_layered_generation_is_deterministic() -> None:
    first = build_layered_candidates(_finalists_fixture(), seed=42)
    second = build_layered_candidates(_finalists_fixture(), seed=42)
    assert first.layer_a.equals(second.layer_a)
    assert first.layer_b.equals(second.layer_b)
    assert first.layer_c.equals(second.layer_c)

