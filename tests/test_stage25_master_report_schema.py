from __future__ import annotations

import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage25.edge_program import run_stage25_master


def _cfg() -> dict:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.setdefault("evaluation", {}).setdefault("stage23", {})["enabled"] = True
    cfg.setdefault("evaluation", {}).setdefault("stage24", {})["enabled"] = False
    cfg.setdefault("risk", {})["max_gross_exposure"] = 1000.0
    return cfg


def test_stage25_master_summary_schema(tmp_path: Path) -> None:
    result = run_stage25_master(
        config=_cfg(),
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        families=["price"],
        composers=["weighted_sum"],
        cost_levels=["realistic"],
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
    for key in (
        "stage",
        "seed",
        "research_run_id",
        "live_run_id",
        "regime_run_id",
        "research",
        "live",
        "live_replay",
        "exit_upgrade",
        "regime_conditional",
        "next_bottleneck",
        "final_verdict",
        "summary_hash",
    ):
        assert key in payload
    assert payload["stage"] == "25"
    assert payload["final_verdict"] in {"NO_EDGE", "WEAK_EDGE"}

