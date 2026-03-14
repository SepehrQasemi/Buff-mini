from __future__ import annotations

import json
from pathlib import Path

from buffmini.stage60 import assess_chain_integrity


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def test_stage60_detects_missing_artifacts(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    runs = tmp_path / "runs"
    run_id = "r1_stage28"
    _write_json(docs / "stage39_signal_generation_summary.json", {"stage": "39", "stage28_run_id": run_id, "summary_hash": "a"})
    _write_json(docs / "stage47_signal_gen2_summary.json", {"stage": "47", "stage28_run_id": run_id, "summary_hash": "b"})
    _write_json(docs / "stage48_tradability_learning_summary.json", {"stage": "48", "stage28_run_id": run_id, "summary_hash": "c"})
    _write_json(docs / "stage52_summary.json", {"stage": "52", "stage28_run_id": run_id, "input_mode": "bootstrap_templates", "summary_hash": "d"})
    _write_json(docs / "stage53_summary.json", {"stage": "53", "stage28_run_id": run_id, "input_mode": "bootstrap_training_dataset", "summary_hash": "e"})
    _write_json(docs / "stage54_summary.json", {"stage": "54", "stage28_run_id": run_id, "summary_hash": "f"})
    _write_json(docs / "stage55_summary.json", {"stage": "55", "stage28_run_id": run_id, "summary_hash": "g"})
    _write_json(docs / "stage56_summary.json", {"stage": "56", "stage28_run_id": run_id, "summary_hash": "h"})
    _write_json(docs / "stage57_summary.json", {"stage": "57", "stage28_run_id": run_id, "summary_hash": "i"})
    out = assess_chain_integrity(docs_dir=docs, runs_dir=runs, budget_mode_selected="validate")
    assert out["status"] == "PARTIAL"
    assert "missing_artifacts:" in out["blocker_reason"]
    assert "bootstrap_forbidden" in out
    assert out["chain_id"]

