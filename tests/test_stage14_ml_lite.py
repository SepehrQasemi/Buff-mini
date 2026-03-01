from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage14.evaluate import (
    run_stage13_14_master_report,
    run_stage14_threshold_calibration,
    run_stage14_weighting,
)


def _cfg() -> dict:
    cfg = deepcopy(load_config(DEFAULT_CONFIG_PATH))
    cfg["evaluation"]["stage14"]["enabled"] = True
    cfg["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 720
    return cfg


def test_stage14_weighting_deterministic(tmp_path: Path) -> None:
    cfg = _cfg()
    a = run_stage14_weighting(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframe="1h",
        runs_root=tmp_path / "runs_a",
        docs_dir=tmp_path / "docs_a",
    )
    b = run_stage14_weighting(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframe="1h",
        runs_root=tmp_path / "runs_b",
        docs_dir=tmp_path / "docs_b",
    )
    assert a["summary"]["model"] == b["summary"]["model"]
    assert float(a["summary"]["l2"]) == float(b["summary"]["l2"])
    assert a["summary"]["classification"] == b["summary"]["classification"]


def test_stage14_threshold_report_and_master_schema(tmp_path: Path) -> None:
    cfg = _cfg()
    run_stage14_threshold_calibration(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframe="1h",
        runs_root=tmp_path / "runs",
        docs_dir=tmp_path / "docs",
    )
    master = run_stage13_14_master_report(
        docs_dir=tmp_path / "docs",
        report_md_name="stage13_14_master_report.md",
        report_json_name="stage13_14_master_summary.json",
    )
    assert "final_verdict" in master
    assert master["final_verdict"] in {"ROBUST_EDGE", "WEAK_EDGE", "NO_EDGE"}
    assert (tmp_path / "docs" / "stage13_14_master_report.md").exists()
    assert (tmp_path / "docs" / "stage13_14_master_summary.json").exists()

