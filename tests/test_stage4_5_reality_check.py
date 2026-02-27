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
