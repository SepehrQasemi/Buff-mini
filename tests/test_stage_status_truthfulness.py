from __future__ import annotations

from buffmini.alpha_v2.reports import truthful_stage_status


def test_trading_stage_with_zero_trades_is_failed() -> None:
    status, failures = truthful_stage_status(
        metrics={"trade_count": 0.0, "walkforward_executed_true_pct": 0.0, "mc_trigger_rate": 0.0},
        status="PASS",
        failures=[],
        stage_type="trading",
        expect_walkforward=False,
        expect_mc=False,
    )
    assert status == "FAILED"
    assert "truth:no_trades_executed" in failures


def test_non_trading_stage_does_not_fail_on_trade_count_zero() -> None:
    status, failures = truthful_stage_status(
        metrics={"trade_count": 0.0},
        status="PASS",
        failures=[],
        stage_type="non_trading",
        expect_walkforward=False,
        expect_mc=False,
    )
    assert status == "PASS"
    assert failures == []


def test_truthful_status_fails_when_expected_wf_or_mc_not_triggered() -> None:
    status, failures = truthful_stage_status(
        metrics={"trade_count": 5.0, "walkforward_executed_true_pct": 0.0, "mc_trigger_rate": 0.0},
        status="PASS",
        failures=[],
        stage_type="trading",
        expect_walkforward=True,
        expect_mc=True,
    )
    assert status == "FAILED"
    assert "truth:walkforward_not_executed" in failures
    assert "truth:mc_not_triggered" in failures
