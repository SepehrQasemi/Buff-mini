from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.forensics.signal_flow import run_signal_flow_trace
from buffmini.stage23.rejects import EXECUTION_REJECT_REASONS


def _stage24_cfg() -> dict:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.setdefault("risk", {})["max_gross_exposure"] = 1000.0

    stage23 = cfg.setdefault("evaluation", {}).setdefault("stage23", {})
    stage23["enabled"] = False

    stage24 = cfg.setdefault("evaluation", {}).setdefault("stage24", {})
    stage24["enabled"] = True
    stage24["base_timeframe"] = "1m"
    stage24["operational_timeframe"] = "1h"
    stage24.setdefault("sizing", {})
    stage24["sizing"]["mode"] = "alloc_pct"
    stage24["sizing"]["alloc_pct"] = 0.25
    stage24["sizing"]["risk_pct_user"] = None
    stage24.setdefault("order_constraints", {})
    stage24["order_constraints"]["min_trade_notional"] = 0.01
    stage24["order_constraints"]["allow_size_bump_to_min_notional"] = True
    stage24["order_constraints"]["max_notional_pct_of_equity"] = 10.0
    stage24.setdefault("simulation", {})
    stage24["simulation"]["initial_equities"] = [1000.0]
    stage24["simulation"]["seed"] = 42
    return cfg


def test_stage24_dry_run_writes_sizing_artifacts(tmp_path: Path) -> None:
    cfg = _stage24_cfg()
    result = run_signal_flow_trace(
        config=cfg,
        seed=42,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        mode="classic",
        stages=["classic"],
        families=["price"],
        composers=["none"],
        max_combos=20,
        dry_run=True,
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
    )
    trace_dir = Path(result["trace_dir"])
    trace_csv = trace_dir / "stage24_sizing_trace.csv"
    summary_json = trace_dir / "stage24_sizing_summary.json"
    assert trace_csv.exists()
    assert summary_json.exists()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    for key in ("valid_count", "invalid_count", "top_invalid_reasons", "notional_min", "notional_max"):
        assert key in summary
    assert int(summary["valid_count"]) >= 1

    trace = pd.read_csv(trace_csv)
    assert not trace.empty
    invalid_reasons = set(
        trace.loc[trace["status"].astype(str) != "VALID", "reason"]
        .astype(str)
        .str.strip()
        .replace("", "UNKNOWN")
        .tolist()
    )
    assert invalid_reasons.issubset(set(EXECUTION_REJECT_REASONS))
