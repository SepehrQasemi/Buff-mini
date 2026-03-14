from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from buffmini.stage61 import materialize_stage57_chain_metrics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def test_stage61_materializes_metrics_from_run_scoped_artifacts(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    runs = tmp_path / "runs"
    run_id = "r2_stage28"
    base = runs / run_id
    (base / "stage48").mkdir(parents=True, exist_ok=True)
    (base / "stage53").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"tradable": 1, "net_return_after_cost": 0.001}, {"tradable": 0, "net_return_after_cost": -0.001}]).to_csv(
        base / "stage48" / "stage48_labels.csv",
        index=False,
    )
    pd.DataFrame([{"candidate_id": "x", "exp_lcb_proxy": 0.01}]).to_csv(base / "stage53" / "stage_b_survivors.csv", index=False)
    _write_json(docs / "stage55_summary.json", {"phase_timings": {"walkforward": 120.0, "monte_carlo": 50.0}, "speedup_projection": {"baseline_runtime_seconds": 1000.0, "optimized_runtime_seconds": 600.0}})
    _write_json(docs / "stage56_summary.json", {"learning_depth": "DEEPENING_MULTI_SIGNAL_MEMORY"})
    _write_json(docs / "stage50_5seed_summary.json", {"executed_seeds": [42, 43, 44]})
    out = materialize_stage57_chain_metrics(
        docs_dir=docs,
        runs_dir=runs,
        stage28_run_id=run_id,
        chain_id="abc",
        config_hash="cfg123",
        data_hash="data123",
        seed=42,
        required_real_sources=[],
    )
    assert out["status"] == "SUCCESS"
    metrics = out["chain_metrics"]
    assert metrics["meta"]["source"] == "stage61_chain_writer_v2"
    assert set(metrics.keys()) >= {"replay_metrics", "walkforward_metrics", "monte_carlo_metrics", "cross_seed_metrics"}
