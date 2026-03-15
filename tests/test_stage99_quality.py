from __future__ import annotations

from buffmini.research.quality import (
    build_quality_transition_rows,
    build_top_k_truth_review,
    explain_quality_reasons,
)


def test_explain_quality_reasons_prefers_meaningful_quality_signals() -> None:
    reasons = explain_quality_reasons(
        {
            "trade_quality_bonus": 0.12,
            "usefulness_prior": 0.08,
            "trade_density_risk": 0.4,
            "clustering_risk": 0.42,
            "thin_evidence_risk": 0.38,
        }
    )
    assert "strong_trade_quality_bonus" in reasons
    assert "useful_trade_density_or_transfer_prior" in reasons
    assert "acceptable_trade_density" in reasons
    assert "lower_clustering_risk" in reasons
    assert "thicker_evidence_profile" in reasons


def test_build_quality_transition_rows_reports_rank_delta_and_reason() -> None:
    before = {
        "evaluations": [
            {"candidate_id": "c1", "family": "continuation", "final_class": "promising_but_unproven"},
        ],
        "ranked_frame": _FakeFrame(
            [
                {
                    "candidate_id": "c1",
                    "rank_score": 0.21,
                    "trade_quality_bonus": 0.02,
                    "trade_density_risk": 0.7,
                }
            ]
        ),
    }
    after = {
        "evaluations": [
            {"candidate_id": "c1", "family": "continuation", "final_class": "promising_but_unproven"},
        ],
        "ranked_frame": _FakeFrame(
            [
                {
                    "candidate_id": "c1",
                    "rank_score": 0.37,
                    "trade_quality_bonus": 0.11,
                    "trade_density_risk": 0.39,
                    "clustering_risk": 0.4,
                    "thin_evidence_risk": 0.43,
                }
            ]
        ),
    }
    rows = build_quality_transition_rows(before=before, after=after)
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "c1"
    assert rows[0]["rank_delta"] == 0.16
    assert "strong_trade_quality_bonus" in rows[0]["change_reason"]


def test_build_top_k_truth_review_exposes_rescue_hints() -> None:
    after = {
        "evaluations": [
            {
                "candidate_id": "c1",
                "family": "continuation",
                "rank_score": 0.52,
                "expected_regime": "trend",
                "first_death_stage": "replay",
                "death_reason": "exp_lcb",
                "transfer_classification": "not_transferable",
                "transfer_diagnostics": ["timing_instability"],
                "monte_carlo_passed": False,
                "robustness_stop_reason": "cost_stress_fail",
                "cost_fragility_risk": 0.6,
                "hold_sanity_risk": 0.2,
                "transfer_risk_prior": 0.4,
            },
            {
                "candidate_id": "c2",
                "family": "breakout",
                "rank_score": 0.48,
                "expected_regime": "compression",
                "first_death_stage": "transfer",
                "death_reason": "transfer_fail",
                "transfer_classification": "not_transferable",
                "transfer_diagnostics": ["regime_mismatch"],
                "monte_carlo_passed": True,
                "robustness_stop_reason": "",
                "cost_fragility_risk": 0.2,
                "hold_sanity_risk": 0.1,
                "transfer_risk_prior": 0.5,
            },
        ]
    }
    rows = build_top_k_truth_review(after=after, top_k=2)
    assert rows[0]["candidate_id"] == "c1"
    assert rows[0]["rescue_hint"] == "lower_cost_exposure"
    assert rows[1]["candidate_id"] == "c2"
    assert rows[1]["rescue_hint"] == "tighten_participation_and_transfer_exposure"


class _FakeFrame:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def to_dict(self, orient: str = "records") -> list[dict[str, object]]:
        assert orient == "records"
        return list(self._rows)
