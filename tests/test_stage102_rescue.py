from __future__ import annotations

from buffmini.research.rescue import classify_rescue_outcome


def test_classify_rescue_outcome_detects_transfer_block() -> None:
    outcome = classify_rescue_outcome(
        base={
            "validation_state": "REAL_REPLAY_READY",
            "first_death_stage": "transfer",
            "transfer_classification": "not_transferable",
        },
        variants=[
            {
                "variant": "diagnostic_transfer_relaxed",
                "diagnostic_only": True,
                "final_class": "promising_but_unproven",
                "first_death_stage": "transfer",
                "replay_exp_lcb": 0.001,
                "robustness_level": 1,
            }
        ],
    )
    assert outcome == "transfer_blocked"


def test_classify_rescue_outcome_detects_generator_weakness() -> None:
    outcome = classify_rescue_outcome(
        base={
            "validation_state": "REAL_REPLAY_READY",
            "first_death_stage": "replay",
            "transfer_classification": "not_transferable",
        },
        variants=[
            {
                "variant": "shorter_hold_horizon",
                "diagnostic_only": False,
                "final_class": "rejected",
                "first_death_stage": "replay",
                "replay_exp_lcb": -0.01,
                "robustness_level": 0,
            },
            {
                "variant": "tighter_invalidation",
                "diagnostic_only": False,
                "final_class": "rejected",
                "first_death_stage": "replay",
                "replay_exp_lcb": -0.005,
                "robustness_level": 0,
            },
        ],
    )
    assert outcome == "still_generator_weak"
