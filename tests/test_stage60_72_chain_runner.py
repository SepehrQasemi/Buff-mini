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


def test_stage60_72_runner_smoke(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    runs = tmp_path / "runs"
    run_id = "r60_stage28"
    base = runs / run_id
    for rel in (
        "stage39/layer_a_candidates.csv",
        "stage47/setup_shortlist.csv",
        "stage48/stage48_ranked_candidates.csv",
        "stage48/stage48_stage_a_survivors.csv",
        "stage48/stage48_stage_b_survivors.csv",
        "stage52/setup_candidates_v2.csv",
        "stage53/predictions.csv",
    ):
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".csv"):
            if "setup_candidates_v2" in rel:
                pd.DataFrame(
                    [
                        {
                            "candidate_id": "s52_1",
                            "source_candidate_id": "s47_1",
                            "family": "structure_pullback_continuation",
                            "timeframe": "1h",
                            "cost_edge_proxy": 0.001,
                            "expected_hold_bars": 8,
                            "tp_before_sl_prob": 0.7,
                            "expected_net_after_cost": 0.001,
                        }
                    ]
                ).to_csv(path, index=False)
            elif "predictions" in rel:
                pd.DataFrame([{"candidate_id": "s52_1", "tp_before_sl_prob": 0.7, "expected_net_after_cost": 0.001}]).to_csv(path, index=False)
            elif "stage_a_survivors" in rel or "stage_b_survivors" in rel:
                pd.DataFrame([{"candidate_id": "s47_1"}]).to_csv(path, index=False)
            elif "stage48_ranked_candidates" in rel:
                pd.DataFrame([{"candidate_id": "s47_1", "rank_score": 0.5, "stage_a_score": 0.5, "layer_score": 0.5, "replay_worthiness": 1}]).to_csv(path, index=False)
            elif "setup_shortlist" in rel:
                pd.DataFrame(
                    [
                        {
                            "candidate_id": "s47_1",
                            "family": "structure_pullback_continuation",
                            "context": "trend",
                            "trigger": "pullback_to_structure_level",
                            "confirmation": "flow_continuation",
                            "invalidation": "structure_break",
                            "modules": "[]",
                            "beam_score": 0.6,
                            "source_branch": "stage47_shortlist",
                        }
                    ]
                ).to_csv(path, index=False)
            elif "layer_a_candidates" in rel:
                pd.DataFrame([{"candidate_id": "s39_1"}]).to_csv(path, index=False)
            else:
                pd.DataFrame([{"candidate_id": "x"}]).to_csv(path, index=False)

    common_summary = {"stage28_run_id": run_id, "summary_hash": "abc"}
    _write_json(docs / "stage39_signal_generation_summary.json", {"stage": "39", **common_summary})
    _write_json(docs / "stage47_signal_gen2_summary.json", {"stage": "47", **common_summary})
    _write_json(docs / "stage48_tradability_learning_summary.json", {"stage": "48", **common_summary})
    _write_json(docs / "stage52_summary.json", {"stage": "52", "input_mode": "stage47_shortlist", **common_summary})
    _write_json(docs / "stage53_summary.json", {"stage": "53", "feature_columns": ["cost_edge_proxy"], "stage_a_survivors": 1, "stage_b_survivors": 1, **common_summary})
    _write_json(docs / "stage54_summary.json", {"stage": "54", **common_summary})
    _write_json(docs / "stage55_summary.json", {"stage": "55", "phase_timings": {"walkforward": 120.0, "monte_carlo": 60.0}, "speedup_projection": {"baseline_runtime_seconds": 1000.0, "optimized_runtime_seconds": 700.0}, **common_summary})
    _write_json(docs / "stage56_summary.json", {"stage": "56", "learning_depth": "EARLY_BUT_STRUCTURED", **common_summary})
    _write_json(docs / "stage57_summary.json", {"stage": "57", "replay_gate": {"passed": False}, "walkforward_gate": {"passed": False}, "monte_carlo_gate": {"passed": False}, "cross_seed_gate": {"passed": False}, **common_summary})
    _write_json(docs / "stage58_summary.json", {"stage": "58", "transfer_result": {"transfer_acceptable": False, "verdict": "PARTIAL"}, **common_summary})
    _write_json(docs / "stage50_5seed_summary.json", {"executed_seeds": [42, 43, 44]})

    cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / "run_stage60_72.py"),
        "--config",
        str(CONFIG_PATH),
        "--runs-dir",
        str(runs),
        "--docs-dir",
        str(docs),
        "--campaign-runs",
        "5",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"run_stage60_72.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    assert (docs / "stage72_summary.json").exists()
