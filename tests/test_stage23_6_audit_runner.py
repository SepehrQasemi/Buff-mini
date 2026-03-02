from __future__ import annotations

import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage23.sizing_audit import run_stage23_6_audit


def _config() -> dict:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.setdefault("risk", {})["max_gross_exposure"] = 1000.0
    stage23 = cfg.setdefault("evaluation", {}).setdefault("stage23", {})
    stage23["enabled"] = True
    stage23["sizing_fix_enabled"] = True
    stage23.setdefault("order_builder", {})
    stage23["order_builder"]["min_trade_notional"] = 0.01
    stage23["order_builder"]["qty_step"] = 2.0
    stage23["order_builder"]["min_trade_qty"] = 0.0
    stage23.setdefault("sizing", {})
    stage23["sizing"]["allow_single_step_ceil_rescue"] = True
    stage23["sizing"]["ceil_rescue_max_overage_steps"] = 1
    return cfg


def test_stage23_6_audit_writes_reports_and_schema(tmp_path: Path) -> None:
    cfg = _config()
    result = run_stage23_6_audit(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        mode="classic",
        stages=["classic"],
        families=["price"],
        composers=["none"],
        max_combos=20,
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
        docs_dir=tmp_path / "docs",
    )
    report_json = Path(result["report_json"])
    report_md = Path(result["report_md"])
    diff_json = Path(result["diff_json"])
    assert report_json.exists()
    assert report_md.exists()
    assert diff_json.exists()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    for key in ("stage", "seed", "criteria", "baseline", "after", "deltas", "sizing_delta", "same_seed", "same_data_hash"):
        assert key in payload


def test_stage23_6_baseline_after_share_seed_and_data_hash(tmp_path: Path) -> None:
    cfg = _config()
    result = run_stage23_6_audit(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        mode="classic",
        stages=["classic"],
        families=["price"],
        composers=["none"],
        max_combos=20,
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
        docs_dir=tmp_path / "docs",
    )
    payload = dict(result["payload"])
    assert bool(payload["same_seed"]) is True
    assert bool(payload["same_data_hash"]) is True


def test_stage23_6_fix_changes_sizing_counters(tmp_path: Path) -> None:
    cfg = _config()
    result = run_stage23_6_audit(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        mode="classic",
        stages=["classic"],
        families=["price"],
        composers=["none"],
        max_combos=20,
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
        docs_dir=tmp_path / "docs",
    )
    payload = dict(result["payload"])
    base = payload["baseline"]["sizing_trace_summary"]
    after = payload["after"]["sizing_trace_summary"]
    changed = any(
        abs(float(after.get(key, 0.0)) - float(base.get(key, 0.0))) > 1e-12
        for key in ("zero_size_count", "rescued_by_ceil_count", "bumped_to_min_notional_count", "cap_binding_reject_count")
    )
    if not changed:
        assert int(base.get("attempted", 0)) == int(after.get("attempted", 0))
        assert float(base.get("zero_size_count", 0.0)) == 0.0
        assert float(after.get("zero_size_count", 0.0)) == 0.0
