"""Stage-5 artifact parser robustness tests."""

from __future__ import annotations

import json

import pandas as pd

from buffmini.ui.components.artifacts import (
    load_generic_curves,
    load_pipeline_summary,
    load_stage3_2_artifacts,
    load_stage3_3_artifacts,
)


def test_parsers_handle_missing_files_without_crashing(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)

    stage32, warnings32 = load_stage3_2_artifacts(run_dir)
    stage33, warnings33 = load_stage3_3_artifacts(run_dir)
    generic, warningsg = load_generic_curves(run_dir)
    pipe, warningsp = load_pipeline_summary(run_dir)

    assert isinstance(stage32, dict)
    assert isinstance(stage33, dict)
    assert isinstance(generic, dict)
    assert isinstance(pipe, dict)
    assert warnings32
    assert warnings33
    assert warningsg
    assert warningsp


def test_parsers_read_minimal_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "r2"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "selector_summary.json").write_text(json.dumps({"run_id": "r2"}), encoding="utf-8")
    pd.DataFrame([{"method": "equal", "leverage": 1, "expected_log_growth": 0.1}]).to_csv(
        run_dir / "selector_table.csv", index=False
    )
    (run_dir / "pipeline_summary.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")

    stage33, warnings33 = load_stage3_3_artifacts(run_dir)
    pipe, warningsp = load_pipeline_summary(run_dir)

    assert stage33["summary"].get("run_id") == "r2"
    assert not stage33["table"].empty
    assert pipe["pipeline_summary"].get("status") == "success"
    assert isinstance(warnings33, list)
    assert isinstance(warningsp, list)
