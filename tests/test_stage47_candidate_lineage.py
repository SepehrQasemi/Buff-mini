from __future__ import annotations

import pandas as pd

from buffmini.stage47.genesis import beam_search_setups, generate_setup_candidates


def test_stage47_candidate_lineage_preserved_after_beam_search() -> None:
    layer_a = pd.DataFrame(
        [
            {"candidate_id": "s39_a", "branch": "branch_a", "broad_context": "trend", "layer_score": 0.6},
            {"candidate_id": "s39_b", "branch": "branch_b", "broad_context": "range", "layer_score": 0.4},
        ]
    )
    setups = generate_setup_candidates(layer_a, seed=42)
    shortlist = beam_search_setups(setups, beam_width=8, per_family_max=4)
    assert not shortlist.empty
    lineage = shortlist.iloc[0]["lineage"]
    assert isinstance(lineage, dict)
    assert "source_candidate_id" in lineage
    assert "source_branch" in lineage
    assert "modules" in lineage

