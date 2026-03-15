from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.family_audit import evaluate_family_audit


def test_stage86_family_audit_builds_inventory_and_blind_spots() -> None:
    cfg = load_config(Path(DEFAULT_CONFIG_PATH))
    audit = evaluate_family_audit(cfg)
    assert audit["family_count"] >= 8
    assert isinstance(audit["blind_spots"], list)
    assert "semantic_overlap_pairs" in audit["overlap_analysis"]
    assert any(int(row.get("context_richness", 0)) >= 2 for row in audit["family_inventory"])
