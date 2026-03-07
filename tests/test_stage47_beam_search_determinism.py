from __future__ import annotations

import pandas as pd

from buffmini.stage47.genesis import beam_search_setups, generate_setup_candidates


def test_stage47_beam_search_is_deterministic() -> None:
    layer_a = pd.DataFrame(
        [
            {"candidate_id": f"s39_{i}", "branch": "b", "broad_context": "shock", "layer_score": 0.3 + i * 0.01}
            for i in range(10)
        ]
    )
    setups = generate_setup_candidates(layer_a, seed=42)
    one = beam_search_setups(setups, beam_width=12, per_family_max=4)
    two = beam_search_setups(setups, beam_width=12, per_family_max=4)
    assert one["candidate_id"].tolist() == two["candidate_id"].tolist()

