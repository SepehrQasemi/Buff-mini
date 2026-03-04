from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.forensics.signal_flow import run_signal_flow_trace


def test_stage28_shadow_live_flags_present(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg_run = json.loads(json.dumps(cfg))
    cfg_run.setdefault("evaluation", {}).setdefault("stage23", {})["enabled"] = True
    cfg_run.setdefault("evaluation", {}).setdefault("stage24", {})["enabled"] = True
    cfg_run.setdefault("evaluation", {}).setdefault("constraints", {})["mode"] = "research"
    cfg_run["evaluation"]["constraints"].setdefault("live", {})
    cfg_run["evaluation"]["constraints"]["live"]["min_trade_notional"] = 1e9

    result = run_signal_flow_trace(
        config=cfg_run,
        seed=42,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        mode="classic",
        stages=["classic"],
        families=["price"],
        composers=["none"],
        max_combos=1,
        dry_run=True,
        runs_root=tmp_path,
        data_dir=tmp_path / "data",
        derived_dir=tmp_path / "derived",
    )
    trace_dir = Path(result["trace_dir"])
    flags_path = trace_dir / "research_infeasible_flags.csv"
    assert flags_path.exists()
    flags = pd.read_csv(flags_path)
    assert "research_infeasible_live" in flags.columns
    assert int(flags.shape[0]) >= 1

