from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _run_script(name: str, *, docs_dir: Path, runs_dir: Path) -> None:
    cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / name),
        "--config",
        str(CONFIG_PATH),
        "--docs-dir",
        str(docs_dir),
        "--runs-dir",
        str(runs_dir),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"{name} failed: {result.stdout}\n{result.stderr}")


def test_stage54_marks_partial_on_dead_stage53_path(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    runs_dir = tmp_path / "runs"
    _write_json(
        docs_dir / "stage53_summary.json",
        {
            "stage28_run_id": "",
            "stage_a_survivors": 0,
            "stage_b_survivors": 0,
        },
    )

    _run_script("run_stage54.py", docs_dir=docs_dir, runs_dir=runs_dir)
    summary = json.loads((docs_dir / "stage54_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PARTIAL"
    assert summary["promotion_timeframes"] == []
    assert summary["final_validation_timeframes"] == []
    assert summary["blocker_reason"] == "dead_stage53_path"


def test_stage53_marks_partial_when_quality_gate_fails(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    runs_dir = tmp_path / "runs"
    stage28_run_id = "r1_stage28"

    stage52_dir = runs_dir / stage28_run_id / "stage52"
    stage48_dir = runs_dir / stage28_run_id / "stage48"
    stage52_dir.mkdir(parents=True, exist_ok=True)
    stage48_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "candidate_id": "s52_x",
                "source_candidate_id": "s47_x",
                "family": "squeeze_flow_breakout",
                "timeframe": "1h",
                "beam_score": 0.5,
                "cost_edge_proxy": 0.001,
                "rr_model": "{'first_target_rr': 1.6}",
                "geometry": "{'stop_distance_pct':0.005,'first_target_pct':0.01,'stretch_target_pct':0.015,'expected_hold_bars':8,'entry_zone':{'low':0.99,'high':1.01}}",
            }
        ]
    ).to_csv(stage52_dir / "setup_candidates_v2.csv", index=False)
    pd.DataFrame([{"candidate_id": "s47_x", "rank_score": 0.6, "replay_worthiness": 1, "stage_a_score": 0.6, "layer_score": 0.7}]).to_csv(
        stage48_dir / "stage48_ranked_candidates.csv",
        index=False,
    )
    pd.DataFrame(columns=["candidate_id"]).to_csv(stage48_dir / "stage48_stage_a_survivors.csv", index=False)
    pd.DataFrame(columns=["candidate_id"]).to_csv(stage48_dir / "stage48_stage_b_survivors.csv", index=False)

    _write_json(docs_dir / "stage52_summary.json", {"stage28_run_id": stage28_run_id})
    _run_script("run_stage53.py", docs_dir=docs_dir, runs_dir=runs_dir)
    summary = json.loads((docs_dir / "stage53_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PARTIAL"
    assert summary["quality_gate_passed"] is False


def test_stage53_handles_missing_stage48_rank_columns_without_crash(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    runs_dir = tmp_path / "runs"
    stage28_run_id = "r2_stage28"

    stage52_dir = runs_dir / stage28_run_id / "stage52"
    stage48_dir = runs_dir / stage28_run_id / "stage48"
    stage52_dir.mkdir(parents=True, exist_ok=True)
    stage48_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "candidate_id": "s52_x",
                "source_candidate_id": "s47_x",
                "family": "squeeze_flow_breakout",
                "timeframe": "1h",
                "beam_score": 0.5,
                "cost_edge_proxy": 0.001,
                "rr_model": "{'first_target_rr': 1.6}",
                "geometry": "{'stop_distance_pct':0.005,'first_target_pct':0.01,'stretch_target_pct':0.015,'expected_hold_bars':8,'entry_zone':{'low':0.99,'high':1.01}}",
            }
        ]
    ).to_csv(stage52_dir / "setup_candidates_v2.csv", index=False)
    pd.DataFrame([{"candidate_id": "s47_x", "rank_score": 0.6}]).to_csv(
        stage48_dir / "stage48_ranked_candidates.csv",
        index=False,
    )
    pd.DataFrame(columns=["candidate_id"]).to_csv(stage48_dir / "stage48_stage_a_survivors.csv", index=False)
    pd.DataFrame(columns=["candidate_id"]).to_csv(stage48_dir / "stage48_stage_b_survivors.csv", index=False)

    _write_json(docs_dir / "stage52_summary.json", {"stage28_run_id": stage28_run_id})
    _run_script("run_stage53.py", docs_dir=docs_dir, runs_dir=runs_dir)
    summary = json.loads((docs_dir / "stage53_summary.json").read_text(encoding="utf-8"))
    assert summary["stage"] == "53"
    assert "status" in summary


def test_stage55_projection_semantics_no_false_measured_target(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    runs_dir = tmp_path / "runs"
    _write_json(
        docs_dir / "stage54_summary.json",
        {
            "stage28_run_id": "",
            "status": "PARTIAL",
        },
    )
    _write_json(
        docs_dir / "stage43_performance_summary.json",
        {
            "baseline": {"runtime_seconds": 200.0},
            "upgraded": {"runtime_seconds": 180.0},
            "phase_runtime_seconds": {"replay_backtest": 100.0},
        },
    )

    _run_script("run_stage55.py", docs_dir=docs_dir, runs_dir=runs_dir)
    summary = json.loads((docs_dir / "stage55_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PARTIAL"
    assert "projected_meets_target" in summary["speedup_projection"]
    assert "meets_stage55_target" not in summary["speedup_projection"]
    assert summary["speedup_measurement"]["measured"] is False


def test_stage56_marks_dead_path_depth(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    runs_dir = tmp_path / "runs"
    _write_json(
        docs_dir / "stage53_summary.json",
        {
            "stage28_run_id": "",
            "stage_a_survivors": 0,
            "stage_b_survivors": 0,
            "quality_gate_passed": False,
        },
    )
    _write_json(docs_dir / "stage55_summary.json", {"status": "PARTIAL"})

    _run_script("run_stage56.py", docs_dir=docs_dir, runs_dir=runs_dir)
    summary = json.loads((docs_dir / "stage56_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PARTIAL"
    assert summary["learning_depth"] == "EARLY_OR_DEAD_PATH"
    assert summary["blocker_reason"] == "dead_upstream_path"


def test_stage57_stays_partial_without_chain_metrics_and_ignores_historical_docs(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    _write_json(docs_dir / "stage52_summary.json", {"status": "SUCCESS"})
    _write_json(
        docs_dir / "stage53_summary.json",
        {
            "status": "SUCCESS",
            "quality_gate_passed": True,
            "stage_a_survivors": 10,
            "stage_b_survivors": 5,
        },
    )
    _write_json(docs_dir / "stage54_summary.json", {"status": "SUCCESS"})
    _write_json(docs_dir / "stage55_summary.json", {"status": "SUCCESS", "projection_only": True})
    _write_json(docs_dir / "stage56_summary.json", {"status": "SUCCESS"})

    _write_json(
        docs_dir / "stage43_performance_summary.json",
        {"upgraded": {"trade_count": 999, "live_best_exp_lcb": 1.0, "wf_executed_pct": 100.0, "mc_trigger_pct": 100.0}},
    )
    _write_json(docs_dir / "stage34_report_summary.json", {"policy": {"max_drawdown": 0.01}})
    _write_json(docs_dir / "stage50_5seed_summary.json", {"skipped": False, "executed_seeds": [1, 2, 3, 4, 5]})

    cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / "run_stage57.py"),
        "--config",
        str(CONFIG_PATH),
        "--docs-dir",
        str(docs_dir),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"run_stage57.py failed: {result.stdout}\n{result.stderr}")

    summary = json.loads((docs_dir / "stage57_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PARTIAL"
    assert summary["verdict"] == "PARTIAL"
    assert str(summary["blocker_reason"]).startswith("missing_chain_metrics_keys:")


def test_stage57_rejects_synthetic_chain_metrics_source(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    _write_json(docs_dir / "stage52_summary.json", {"status": "SUCCESS"})
    _write_json(
        docs_dir / "stage53_summary.json",
        {
            "status": "SUCCESS",
            "quality_gate_passed": True,
            "stage_a_survivors": 10,
            "stage_b_survivors": 5,
        },
    )
    _write_json(docs_dir / "stage54_summary.json", {"status": "SUCCESS"})
    _write_json(docs_dir / "stage55_summary.json", {"status": "SUCCESS"})
    _write_json(docs_dir / "stage56_summary.json", {"status": "SUCCESS"})
    _write_json(
        docs_dir / "stage57_chain_metrics.json",
        {
            "replay_metrics": {"trade_count": 40, "exp_lcb": 0.01, "maxDD": 0.15, "failure_reason_dominance": 0.4},
            "walkforward_metrics": {"usable_windows": 5, "median_forward_exp_lcb": 0.002},
            "monte_carlo_metrics": {"conservative_downside_bound": 0.001},
            "cross_seed_metrics": {"surviving_seeds": 3},
            "meta": {"source": "stage53_survivor_artifacts"},
        },
    )

    cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / "run_stage57.py"),
        "--config",
        str(CONFIG_PATH),
        "--docs-dir",
        str(docs_dir),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"run_stage57.py failed: {result.stdout}\n{result.stderr}")

    summary = json.loads((docs_dir / "stage57_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PARTIAL"
    assert summary["verdict"] == "PARTIAL"
    assert summary["blocker_reason"] == "invalid_chain_metrics_source:stage53_survivor_artifacts"
