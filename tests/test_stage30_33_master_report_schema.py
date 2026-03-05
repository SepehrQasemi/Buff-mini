from __future__ import annotations

from buffmini.stage33.master import build_master_summary


def test_stage30_33_master_report_schema() -> None:
    summary = build_master_summary(
        head_commit="abc",
        run_ids={"stage30": "r1", "stage31": "r2", "stage32": "r3", "stage33": "r4"},
        stage30={"status": "COMPLETE"},
        stage31={"status": "COMPLETE"},
        stage32={"wf_executed_pct": 50.0, "mc_trigger_pct": 30.0},
        stage33={"policy_metrics": {"research": {"exp_lcb": 0.01}, "live": {"exp_lcb": 0.0}}},
        drift={"warnings": []},
        config_hash="cfg",
        data_hash="data",
        resolved_end_ts="2026-01-01T00:00:00+00:00",
        runtime_seconds=10.0,
    )
    required = {
        "stage",
        "head_commit",
        "run_ids",
        "config_hash",
        "data_hash",
        "resolved_end_ts",
        "stage30",
        "stage31",
        "stage32",
        "stage33",
        "drift",
        "verdict",
        "next_bottleneck",
        "runtime_seconds",
    }
    assert required.issubset(set(summary.keys()))

