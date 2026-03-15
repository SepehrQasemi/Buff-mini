from __future__ import annotations

from buffmini.research.transfer import build_transfer_intelligence


def test_stage92_transfer_intelligence_builds_regime_map() -> None:
    intelligence = build_transfer_intelligence(
        [
            {"classification": "transferable", "diagnostics": ["trigger_rarity"], "expected_regime": "trend"},
            {"classification": "not_transferable", "diagnostics": ["regime_mismatch"], "expected_regime": "trend"},
            {"classification": "partially_transferable", "diagnostics": ["cost_collapse"], "expected_regime": "compression"},
        ]
    )
    assert intelligence["transfer_class_counts"]["transferable"] == 1
    assert intelligence["failure_diagnostics"]["regime_mismatch"] == 1
    assert any(row["regime"] == "trend" for row in intelligence["regime_portability_map"])
