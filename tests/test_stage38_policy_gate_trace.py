from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from buffmini.stage38.audit import build_lineage_table_from_stage28
from buffmini.stage38.trace import build_stage28_execution_trace


def _build_stage28_fixture(tmp_path: Path) -> Path:
    stage28_dir = tmp_path / "stage28"
    stage28_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"candidate_id": "c1"}, {"candidate_id": "c2"}]).to_csv(stage28_dir / "selected_candidates_stageA.csv", index=False)
    pd.DataFrame([{"candidate_id": "c1"}]).to_csv(stage28_dir / "selected_candidates_stageB.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "candidate": "Dummy",
                "family": "flow",
                "context": "RANGE",
                "exp_lcb": 0.01,
            }
        ]
    ).to_csv(stage28_dir / "finalists_stageC.csv", index=False)
    pd.DataFrame([{"candidate_id": "c1", "context_occurrences": 10}]).to_csv(stage28_dir / "context_matrix.csv", index=False)
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "context": "RANGE",
                "net_score": 0.10,
                "active_candidates": "c1",
                "final_signal": 1,
            },
            {
                "timestamp": "2026-01-01T01:00:00Z",
                "context": "RANGE",
                "net_score": 0.20,
                "active_candidates": "c1",
                "final_signal": -1,
            },
        ]
    ).to_csv(stage28_dir / "policy_trace.csv", index=False)
    pd.DataFrame(columns=["timestamp", "context", "reason"]).to_csv(stage28_dir / "shadow_live_rejects.csv", index=False)
    pd.DataFrame([{"equity": 1000, "feasible_pct": 0.9}]).to_csv(stage28_dir / "feasibility_envelope.csv", index=False)
    pd.DataFrame([{"wf_triggered": True, "mc_triggered": True}]).to_csv(stage28_dir / "usability_trace.csv", index=False)

    policy = {
        "policy_id": "p1",
        "conflict_mode": "net",
        "contexts": {
            "RANGE": {
                "status": "OK",
                "candidates": [{"candidate_id": "c1", "weight": 1.0}],
            }
        },
        "warnings": [],
    }
    (stage28_dir / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (stage28_dir / "policy_spec.md").write_text("# policy\n", encoding="utf-8")

    summary = {
        "run_id": "fixture_stage28",
        "seed": 42,
        "mode": "both",
        "timeframes": ["1h"],
        "used_symbols": ["BTC/USDT"],
        "config_hash": "cfg_hash",
        "data_snapshot_id": "snap_id",
        "data_snapshot_hash": "snap_hash",
        "coverage_gate_status": "PASS",
        "qualified_finalists": 1,
        "wf_executed_pct": 100.0,
        "mc_trigger_pct": 100.0,
        "policy_metrics": {
            "research": {"trade_count": 2.0, "exp_lcb": 0.01},
            "live": {"trade_count": 2.0, "exp_lcb": 0.01},
        },
        "shadow_live_reject_rate": 0.0,
        "shadow_live_top_reasons": {},
        "next_bottleneck": "none",
        "verdict": "WEAK_EDGE",
    }
    (stage28_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return stage28_dir


def test_stage38_policy_gate_trace_reports_expected_counts(tmp_path: Path) -> None:
    stage28_dir = _build_stage28_fixture(tmp_path)
    payload = build_stage28_execution_trace(stage28_dir=stage28_dir, config_path=Path("configs/default.yaml"))
    events = list(payload.get("trace_events", []))
    composer = next(event for event in events if event.get("stage") == "composer")
    details = dict(composer.get("details", {}))
    assert int(details.get("candidate_rows_active", 0)) == 2
    assert int(details.get("net_score_nonzero_rows", 0)) == 2
    assert int(details.get("final_signal_nonzero_rows", 0)) == 2

    lineage = build_lineage_table_from_stage28(stage28_dir=stage28_dir, threshold=0.0, quality_floor=-0.02)
    assert int(lineage["composer_signal_count"]) == int(lineage["engine_raw_signal_count"]) == 2
    assert bool(lineage["contradiction_fixed"]) is True

