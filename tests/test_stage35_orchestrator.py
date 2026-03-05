from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from buffmini.config import load_config
from scripts.run_stage35 import run_stage35_pipeline


def _write_cfg(tmp_path: Path, *, coinapi_enabled: bool) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    base = load_config(repo_root / "configs" / "default.yaml")
    base["coinapi"]["enabled"] = bool(coinapi_enabled)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return cfg_path


def test_stage35_requires_key_when_enabled_and_not_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_cfg(tmp_path, coinapi_enabled=True)
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COINAPI_KEY", raising=False)
    with pytest.raises(SystemExit, match="COINAPI_KEY is required"):
        run_stage35_pipeline(
            config_path=cfg,
            seed=42,
            dry_run=False,
            runs_dir=tmp_path / "runs",
            allow_insufficient_coverage=False,
        )


def test_stage35_dry_run_planner_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_cfg(tmp_path, coinapi_enabled=True)
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    out = run_stage35_pipeline(
        config_path=cfg,
        seed=42,
        dry_run=True,
        runs_dir=tmp_path / "runs",
        allow_insufficient_coverage=False,
    )
    assert str(out["run_id"]).endswith("_stage35")
    assert "plan_counts" in out
    assert (tmp_path / "docs" / "stage35_report_summary.json").exists()
    assert (tmp_path / "docs" / "stage35_coinapi_usage_summary.json").exists()


def test_stage35_report_schema_keys_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_cfg(tmp_path, coinapi_enabled=False)
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    out = run_stage35_pipeline(
        config_path=cfg,
        seed=42,
        dry_run=True,
        runs_dir=tmp_path / "runs",
        allow_insufficient_coverage=True,
    )
    required = {
        "head_commit",
        "run_id",
        "seed",
        "coinapi_enabled",
        "plan_id",
        "plan_counts",
        "usage",
        "coverage",
        "status",
        "ml",
        "storage",
    }
    assert required.issubset(set(out.keys()))

