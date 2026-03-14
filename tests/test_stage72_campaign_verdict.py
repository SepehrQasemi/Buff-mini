from __future__ import annotations

from buffmini.stage72 import derive_free_campaign_verdict


def test_stage72_emits_no_edge_after_fail_streak() -> None:
    out = derive_free_campaign_verdict(
        replay_gate={"passed": False},
        walkforward_gate={"passed": False},
        monte_carlo_gate={"passed": False},
        cross_seed_gate={"passed": False},
        transfer_acceptable=False,
        transfer_required=True,
        campaign_runs=20,
        frozen_scope_fail_streak=3,
        scope_frozen=True,
    )
    assert out["verdict"] == "NO_EDGE_IN_SCOPE"


def test_stage72_emits_medium_edge_when_all_pass_with_transfer() -> None:
    out = derive_free_campaign_verdict(
        replay_gate={"passed": True},
        walkforward_gate={"passed": True},
        monte_carlo_gate={"passed": True},
        cross_seed_gate={"passed": True},
        transfer_acceptable=True,
        transfer_required=True,
        campaign_runs=20,
        frozen_scope_fail_streak=0,
        scope_frozen=True,
    )
    assert out["verdict"] == "MEDIUM_EDGE"


def test_stage72_blocks_positive_verdict_when_transfer_required_but_missing() -> None:
    out = derive_free_campaign_verdict(
        replay_gate={"passed": True},
        walkforward_gate={"passed": True},
        monte_carlo_gate={"passed": True},
        cross_seed_gate={"passed": True},
        transfer_acceptable=False,
        transfer_required=True,
        campaign_runs=20,
        frozen_scope_fail_streak=0,
        scope_frozen=True,
    )
    assert out["verdict"] == "PARTIAL"


def test_stage72_allows_weak_edge_only_when_transfer_not_required() -> None:
    out = derive_free_campaign_verdict(
        replay_gate={"passed": True},
        walkforward_gate={"passed": True},
        monte_carlo_gate={"passed": True},
        cross_seed_gate={"passed": True},
        transfer_acceptable=False,
        transfer_required=False,
        campaign_runs=20,
        frozen_scope_fail_streak=0,
        scope_frozen=False,
    )
    assert out["verdict"] == "WEAK_EDGE"


def test_stage72_does_not_emit_no_edge_for_exploratory_scope() -> None:
    out = derive_free_campaign_verdict(
        replay_gate={"passed": False},
        walkforward_gate={"passed": False},
        monte_carlo_gate={"passed": False},
        cross_seed_gate={"passed": False},
        transfer_acceptable=False,
        transfer_required=True,
        campaign_runs=20,
        frozen_scope_fail_streak=3,
        scope_frozen=False,
    )
    assert out["verdict"] == "PARTIAL"
