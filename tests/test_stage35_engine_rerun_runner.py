from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts import run_stage35_engine_rerun


def _args(tmp_path: Path) -> Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    return Namespace(
        config=repo_root / "configs" / "default.yaml",
        seed=42,
        runs_dir=tmp_path / "runs",
    )


def test_engine_rerun_skips_when_master_download_summary_missing(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    payload = run_stage35_engine_rerun.run_engine_rerun(_args(tmp_path))
    assert payload["status"] == "NOT_EXECUTED"
    assert payload["reason"] == "stage35_7_report_summary_missing"
    assert (tmp_path / "docs" / "stage35_7_engine_rerun_summary.json").exists()


def test_engine_rerun_skips_when_coverage_not_ok(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "stage35_7_report_summary.json").write_text(
        json.dumps({"status": "INSUFFICIENT_COVERAGE", "coverage_ok": False}, indent=2),
        encoding="utf-8",
    )
    payload = run_stage35_engine_rerun.run_engine_rerun(_args(tmp_path))
    assert payload["status"] == "NOT_EXECUTED"
    assert payload["reason"] == "coverage_insufficient_or_download_blocked"

