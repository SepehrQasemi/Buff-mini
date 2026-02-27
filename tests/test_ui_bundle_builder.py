"""Tests for Stage-5 ui_bundle builder."""

from __future__ import annotations

import json

import pandas as pd

from buffmini.ui_bundle import builder, discover


def test_ui_bundle_builder_creates_required_outputs(tmp_path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    pipeline_run = runs_dir / "pipeline_1"
    pipeline_run.mkdir(parents=True, exist_ok=True)

    stage2_run = runs_dir / "stage2_a"
    stage2_run.mkdir(parents=True, exist_ok=True)
    stage3_run = runs_dir / "stage3_3_a"
    stage3_run.mkdir(parents=True, exist_ok=True)
    stage4_sim_run = runs_dir / "stage4_sim_a"
    stage4_sim_run.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "window": ["holdout", "holdout"],
            "portfolio_return": [0.0, 0.01],
            "equity": [10000.0, 10100.0],
            "exposure": [0.2, 0.3],
        }
    ).to_csv(stage2_run / "portfolio_equal_weight.csv", index=False)

    (stage2_run / "portfolio_report.md").write_text("# Stage2\n", encoding="utf-8")

    (stage3_run / "selector_summary.json").write_text(
        json.dumps(
            {
                "overall_choice": {
                    "method": "equal",
                    "chosen_leverage": 2.0,
                    "chosen_row": {"p_ruin": 0.01, "expected_log_growth": 0.02},
                }
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "symbol": ["BTC/USDT"],
            "direction": [1],
        }
    ).to_csv(stage4_sim_run / "orders.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "trigger": ["daily_loss"],
        }
    ).to_csv(stage4_sim_run / "killswitch_events.csv", index=False)

    (pipeline_run / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "run_id": "pipeline_1",
                "status": "success",
                "config_hash": "cfg123",
                "data_hash": "data123",
                "stage1_run_id": "stage1_a",
                "stage2_run_id": "stage2_a",
                "stage3_3_run_id": "stage3_3_a",
                "stage4_sim_run_id": "stage4_sim_a",
                "chosen_method": "equal",
                "chosen_leverage": 2.0,
                "reports": {
                    "stage2_report": str(stage2_run / "portfolio_report.md"),
                },
            }
        ),
        encoding="utf-8",
    )

    (pipeline_run / "pipeline_config.yaml").write_text(
        """
universe:
  symbols: [BTC/USDT, ETH/USDT]
  timeframe: 1h
search:
  seed: 42
evaluation:
  stage1:
    holdout_months: 12
execution:
  mode: net
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(builder, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(discover, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(discover, "PROJECT_ROOT", tmp_path)

    builder.build_ui_bundle_from_pipeline(pipeline_run)

    bundle_dir = pipeline_run / "ui_bundle"
    assert (bundle_dir / "summary_ui.json").exists()
    assert (bundle_dir / "reports_index.json").exists()
    assert (bundle_dir / "charts_index.json").exists()

    summary = json.loads((bundle_dir / "summary_ui.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == "pipeline_1"
    assert summary["chosen_method"] == "equal"


def test_ui_bundle_builder_handles_missing_sources_with_warnings(tmp_path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    pipeline_run = runs_dir / "pipeline_missing"
    pipeline_run.mkdir(parents=True, exist_ok=True)

    (pipeline_run / "pipeline_summary.json").write_text(
        json.dumps({"run_id": "pipeline_missing", "status": "failed", "error": "test"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(builder, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(discover, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(discover, "PROJECT_ROOT", tmp_path)

    builder.build_ui_bundle_from_pipeline(pipeline_run)

    summary = json.loads((pipeline_run / "ui_bundle" / "summary_ui.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert isinstance(summary["bundle_build_warnings"], list)
    assert summary["bundle_build_warnings"]


def test_pipeline_requires_stage1_lineage(tmp_path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    pipeline_run = runs_dir / "pipeline_success_missing_lineage"
    pipeline_run.mkdir(parents=True, exist_ok=True)

    (pipeline_run / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "run_id": "pipeline_success_missing_lineage",
                "status": "success",
                "stage2_run_id": "stage2_a",
                "stage3_3_run_id": "stage3_3_a",
            }
        ),
        encoding="utf-8",
    )
    (pipeline_run / "pipeline_config.yaml").write_text(
        """
universe:
  symbols: [BTC/USDT]
  timeframe: 1h
search:
  seed: 42
evaluation:
  stage1:
    holdout_months: 12
execution:
  mode: net
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(builder, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(discover, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(discover, "PROJECT_ROOT", tmp_path)

    try:
        builder.build_ui_bundle_from_pipeline(pipeline_run)
    except ValueError as exc:
        assert "stage1_run_id" in str(exc)
    else:
        raise AssertionError("Expected lineage validation failure for missing stage1_run_id")
