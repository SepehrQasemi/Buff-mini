from __future__ import annotations

import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage25.edge_program import run_stage25b_edge_program


def _cfg() -> dict:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.setdefault("evaluation", {}).setdefault("stage23", {})["enabled"] = True
    cfg.setdefault("evaluation", {}).setdefault("stage24", {})["enabled"] = False
    cfg.setdefault("risk", {})["max_gross_exposure"] = 1000.0
    return cfg


def test_stage25b_runner_outputs_and_summary_schema(tmp_path: Path) -> None:
    result = run_stage25b_edge_program(
        config=_cfg(),
        seed=42,
        dry_run=True,
        mode="research",
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
    assert Path(result["results_csv"]).exists()
    assert Path(result["results_json"]).exists()
    assert Path(result["best_candidates_json"]).exists()
    assert Path(result["report_md"]).exists()
    assert Path(result["report_json"]).exists()

    payload = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
    for key in (
        "stage",
        "run_id",
        "seed",
        "mode",
        "symbols",
        "timeframes",
        "families",
        "cost_levels",
        "metrics",
        "best_candidates",
        "status",
        "summary_hash",
    ):
        assert key in payload
    assert payload["stage"] == "25B"
    assert payload["mode"] == "research"

