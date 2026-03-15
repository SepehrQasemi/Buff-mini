from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.diagnostics import compute_near_miss_distance, resolve_first_death_stage
from buffmini.research.funnel import diagnose_funnel_pressure


def test_stage88_funnel_pressure_diagnosis_identifies_near_miss_pressure() -> None:
    diagnosis = diagnose_funnel_pressure(
        evaluations=[
            {"candidate_hierarchy": "promising_but_unproven", "near_miss_distance": 0.22},
            {"candidate_hierarchy": "interesting_but_fragile", "near_miss_distance": 0.40},
        ],
        blocked_count=0,
        candidate_count=20,
    )
    assert diagnosis["dominant_culprit"] == "funnel_pressure_over_tight"


def test_stage88_skipped_transfer_is_neutral_in_survival_diagnostics() -> None:
    cfg = load_config(Path(DEFAULT_CONFIG_PATH))
    neutral = compute_near_miss_distance(
        replay_trade_count=50,
        replay_exp_lcb=0.01,
        walkforward_usable_windows=3,
        robustness_level=2,
        transfer_classification="not_evaluated",
        config=cfg,
    )
    penalized = compute_near_miss_distance(
        replay_trade_count=50,
        replay_exp_lcb=0.01,
        walkforward_usable_windows=3,
        robustness_level=2,
        transfer_classification="not_transferable",
        config=cfg,
    )
    assert neutral < penalized
    death_stage, death_reason = resolve_first_death_stage(
        replay_trade_count=50,
        replay_exp_lcb=0.01,
        replay_allowed=True,
        walkforward_usable_windows=6,
        monte_carlo_passed=True,
        perturbation_passed=True,
        split_passed=True,
        transfer_classification="not_evaluated",
        market_blocked_reason="",
        config=cfg,
    )
    assert death_stage == "survived"
    assert death_reason == ""
