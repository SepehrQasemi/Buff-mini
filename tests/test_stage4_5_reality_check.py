"""Stage-4.5 reality-check tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from buffmini.execution.reality_check import RealityCheckConfig, run_reality_check


def _setup_run(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    run_dir = runs / "pipeline_test"
    bundle = run_dir / "ui_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    stage4_run = runs / "stage4_run_test"
    stage4_run.mkdir(parents=True, exist_ok=True)

    policy_snapshot = {
        "selected_method": "equal",
        "leverage": 1.0,
        "execution_mode": "net",
        "caps": {
            "max_gross_exposure": 5.0,
            "max_net_exposure_per_symbol": 5.0,
            "max_open_positions": 10,
        },
        "costs": {
            "round_trip_cost_pct": 0.1,
            "slippage_pct": 0.0005,
            "funding_pct_per_day": 0.0,
        },
        "kill_switch": {
            "enabled": True,
            "max_daily_loss_pct": 5.0,
            "max_peak_to_valley_dd_pct": 20.0,
            "max_consecutive_losses": 8,
            "cool_down_bars": 48,
        },
    }
    (stage4_run / "policy_snapshot.json").write_text(
        json.dumps(policy_snapshot),
        encoding="utf-8",
    )

    (run_dir / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "run_id": "pipeline_test",
                "status": "success",
                "stage1_run_id": "stage1_test",
                "stage2_run_id": "stage2_test",
                "stage3_3_run_id": "stage3_test",
                "stage4_run_id": "stage4_run_test",
            }
        ),
        encoding="utf-8",
    )

    (bundle / "summary_ui.json").write_text(
        json.dumps(
            {
                "run_id": "pipeline_test",
                "status": "success",
                "symbols": ["BTC/USDT"],
                "timeframe": "1h",
                "chosen_method": "equal",
                "chosen_leverage": 1.0,
                "execution_mode": "net",
                "key_metrics": {"pf": 1.1, "maxdd": 0.1, "p_ruin": 0.01, "expected_log_growth": 0.02},
                "config_hash": "cfg",
                "data_hash": "data",
                "seed": 42,
                "stages": {"stage4_run_id": "stage4_run_test"},
                "policy_snapshot": policy_snapshot,
            }
        ),
        encoding="utf-8",
    )

    ts = pd.date_range("2026-01-01", periods=24 * 28, freq="h", tz="UTC")
    rets = pd.Series([0.0005 if i % 5 else -0.0002 for i in range(len(ts))], dtype=float)
    eq = (1.0 + rets).cumprod() * 10000
    pd.DataFrame({"timestamp": ts, "equity": eq}).to_csv(bundle / "equity_curve.csv", index=False)
    pd.DataFrame({"timestamp": ts, "exposure": [0.5] * len(ts)}).to_csv(bundle / "exposure.csv", index=False)
    pd.DataFrame({"timestamp": ts[::12], "symbol": ["BTC/USDT"] * len(ts[::12])}).to_csv(bundle / "trades.csv", index=False)

    return run_dir


def test_reality_check_outputs_and_required_keys(tmp_path: Path) -> None:
    run_dir = _setup_run(tmp_path)
    rc_dir = run_reality_check(run_id=run_dir.name, runs_dir=tmp_path / "runs", cfg=RealityCheckConfig(seed=7))

    assert (rc_dir / "reality_check_summary.json").exists()
    assert (rc_dir / "rolling_forward_steps.csv").exists()
    assert (rc_dir / "perturbation_table.csv").exists()
    assert (rc_dir / "execution_drag_table.csv").exists()

    summary = json.loads((rc_dir / "reality_check_summary.json").read_text(encoding="utf-8"))
    for key in ["confidence_score", "verdict", "reasons", "rolling_forward", "perturbation", "execution_drag"]:
        assert key in summary
    assert 0.0 <= float(summary["confidence_score"]) <= 1.0
    assert summary["verdict"] in {"PASS", "WARN", "FAIL"}


def test_reality_check_is_deterministic(tmp_path: Path) -> None:
    run_dir = _setup_run(tmp_path)
    rc_dir_1 = run_reality_check(run_id=run_dir.name, runs_dir=tmp_path / "runs", cfg=RealityCheckConfig(seed=99))
    first = (rc_dir_1 / "reality_check_summary.json").read_bytes()

    rc_dir_2 = run_reality_check(run_id=run_dir.name, runs_dir=tmp_path / "runs", cfg=RealityCheckConfig(seed=99))
    second = (rc_dir_2 / "reality_check_summary.json").read_bytes()

    assert first == second


def test_reality_check_policy_mismatch_detected(tmp_path: Path) -> None:
    run_dir = _setup_run(tmp_path)
    bundle_summary_path = run_dir / "ui_bundle" / "summary_ui.json"
    summary = json.loads(bundle_summary_path.read_text(encoding="utf-8"))
    summary["policy_snapshot"]["leverage"] = 2.0
    bundle_summary_path.write_text(json.dumps(summary), encoding="utf-8")

    try:
        run_reality_check(run_id=run_dir.name, runs_dir=tmp_path / "runs", cfg=RealityCheckConfig(seed=42))
    except ValueError as exc:
        assert "policy snapshot mismatch" in str(exc)
    else:
        raise AssertionError("Expected policy mismatch to raise ValueError")
