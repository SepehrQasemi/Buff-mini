from __future__ import annotations

import pandas as pd

from buffmini.stage47.genesis import generate_setup_candidates, validate_setup_candidate


def test_stage47_setup_schema_fields_present() -> None:
    layer_a = pd.DataFrame(
        [
            {
                "candidate_id": "s39_a",
                "branch": "vol_compression_flow_burst",
                "broad_context": "shock",
                "layer_score": 0.5,
            }
        ]
    )
    out = generate_setup_candidates(layer_a, seed=42)
    assert not out.empty
    row = dict(out.iloc[0].to_dict())
    validate_setup_candidate(row)
    for key in ("context", "trigger", "confirmation", "invalidation"):
        assert str(row.get(key, "")).strip()

