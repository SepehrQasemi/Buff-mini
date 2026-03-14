from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def test_stage53_writes_replay_metrics_real(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    runs = tmp_path / "runs"
    run_id = "r53_test"
    stage52_dir = runs / run_id / "stage52"
    stage48_dir = runs / run_id / "stage48"
    stage52_dir.mkdir(parents=True, exist_ok=True)
    stage48_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "candidate_id": "s52_1",
                "source_candidate_id": "s47_1",
                "family": "structure_pullback_continuation",
                "timeframe": "1h",
                "beam_score": 0.8,
                "cost_edge_proxy": 0.001,
                "rr_model": "{'first_target_rr': 1.8}",
                "geometry": "{'expected_hold_bars': 8, 'stop_distance_pct': 0.005, 'first_target_pct': 0.01, 'stretch_target_pct': 0.015, 'entry_zone': {'low': 0.99, 'high': 1.01}}",
                "expected_hold_bars": 8,
                "exp_lcb_proxy": 0.001,
            }
        ]
    ).to_csv(stage52_dir / "setup_candidates_v2.csv", index=False)
    pd.DataFrame([{"candidate_id": "s47_1", "rank_score": 1.0, "stage_a_score": 1.0, "layer_score": 1.0, "replay_worthiness": 1}]).to_csv(
        stage48_dir / "stage48_ranked_candidates.csv",
        index=False,
    )
    pd.DataFrame([{"candidate_id": "s47_1"}]).to_csv(stage48_dir / "stage48_stage_a_survivors.csv", index=False)
    pd.DataFrame([{"candidate_id": "s47_1"}]).to_csv(stage48_dir / "stage48_stage_b_survivors.csv", index=False)
    pd.DataFrame([{"tradable": 1, "net_return_after_cost": 0.001}]).to_csv(stage48_dir / "stage48_labels.csv", index=False)

    docs.mkdir(parents=True, exist_ok=True)
    (docs / "stage52_summary.json").write_text(json.dumps({"stage": "52", "stage28_run_id": run_id}, indent=2), encoding="utf-8")
    (docs / "stage39_signal_generation_summary.json").write_text(json.dumps({"stage28_run_id": run_id}, indent=2), encoding="utf-8")

    cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / "run_stage53.py"),
        "--config",
        str(CONFIG_PATH),
        "--runs-dir",
        str(runs),
        "--docs-dir",
        str(docs),
        "--stage28-run-id",
        run_id,
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"run_stage53.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    assert (runs / run_id / "stage53" / "replay_metrics_real.json").exists()
    summary = json.loads((docs / "stage53_summary.json").read_text(encoding="utf-8"))
    assert summary["metric_source_type"] == "real_replay"
    assert summary["validated_candidate_id"] == "s52_1"
