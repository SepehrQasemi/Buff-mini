from __future__ import annotations

import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage24.audit import run_stage24_audit


def _cfg() -> dict:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.setdefault("risk", {})["max_gross_exposure"] = 1000.0
    cfg.setdefault("evaluation", {}).setdefault("stage23", {})["enabled"] = False
    return cfg


def test_stage24_report_schema_keys(tmp_path: Path) -> None:
    result = run_stage24_audit(
        config=_cfg(),
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        base_timeframe="1m",
        operational_timeframe="1h",
        initial_equities=[100.0, 1000.0],
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
        docs_dir=tmp_path / "docs",
    )
    report_json = Path(result["report_json"])
    report_md = Path(result["report_md"])
    assert report_json.exists()
    assert report_md.exists()

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    for key in ("baseline", "risk_pct_mode", "alloc_pct_mode", "capital_sim", "verdict", "next_bottleneck"):
        assert key in payload
    assert bool(payload.get("same_seed_all_modes", False)) is True
    assert bool(payload.get("same_data_hash_all_modes", False)) is True
