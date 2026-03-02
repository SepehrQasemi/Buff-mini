from __future__ import annotations

import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage23.audit import run_stage23_audit


def _config() -> dict:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.setdefault("evaluation", {}).setdefault("stage23", {}).setdefault("eligibility", {})["min_score_default"] = 0.85
    return cfg


def test_stage23_disabled_vs_enabled_trace_differs(tmp_path: Path) -> None:
    cfg = _config()
    result = run_stage23_audit(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        mode="v2",
        stages=["15"],
        families=["price"],
        composers=["none"],
        max_combos=10,
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
        docs_dir=tmp_path / "docs",
    )
    payload = dict(result["payload"])
    baseline = payload["baseline"]["metrics"]
    after = payload["after"]["metrics"]
    changed = any(abs(float(after[key]) - float(baseline[key])) > 1e-12 for key in baseline.keys())
    assert changed


def test_stage23_report_schema_contract(tmp_path: Path) -> None:
    cfg = _config()
    result = run_stage23_audit(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        mode="classic",
        stages=["classic"],
        families=["price"],
        composers=["none"],
        max_combos=10,
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
        docs_dir=tmp_path / "docs",
    )
    report_json = Path(result["report_json"])
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    for key in ("stage", "seed", "baseline", "after", "deltas", "biggest_remaining_bottleneck", "same_seed", "same_data_hash"):
        assert key in payload


def test_stage23_baseline_after_share_seed_and_data_hash(tmp_path: Path) -> None:
    cfg = _config()
    result = run_stage23_audit(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        mode="v2",
        stages=["15"],
        families=["price"],
        composers=["none"],
        max_combos=10,
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
        docs_dir=tmp_path / "docs",
    )
    payload = dict(result["payload"])
    assert bool(payload["same_seed"]) is True
    assert bool(payload["same_data_hash"]) is True
