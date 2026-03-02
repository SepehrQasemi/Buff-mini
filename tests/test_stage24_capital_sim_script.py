from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage24.capital_sim import run_stage24_capital_sim


def _cfg() -> dict:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.setdefault("evaluation", {}).setdefault("stage23", {})["enabled"] = False
    cfg.setdefault("evaluation", {}).setdefault("stage24", {})
    cfg.setdefault("risk", {})["max_gross_exposure"] = 1000.0
    return cfg


def test_stage24_capital_sim_outputs_and_schema(tmp_path: Path) -> None:
    result = run_stage24_capital_sim(
        config=_cfg(),
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        base_timeframe="1m",
        operational_timeframe="1h",
        mode="risk_pct",
        initial_equities=[100.0, 1000.0],
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
        docs_dir=tmp_path / "docs",
    )
    results_csv = Path(result["results_csv"])
    results_json = Path(result["results_json"])
    capital_doc = Path(result["capital_doc"])
    assert results_csv.exists()
    assert results_json.exists()
    assert capital_doc.exists()

    frame = pd.read_csv(results_csv)
    assert list(frame["initial_equity"]) == [100.0, 1000.0]
    for col in (
        "final_equity",
        "return_pct",
        "max_drawdown",
        "trade_count",
        "avg_notional",
        "avg_risk_pct_used",
        "invalid_order_pct",
        "top_invalid_reason",
    ):
        assert col in frame.columns

    payload = json.loads(results_json.read_text(encoding="utf-8"))
    for key in ("stage", "run_id", "seed", "rows", "results_hash", "runtime_seconds"):
        assert key in payload


def test_stage24_capital_sim_deterministic_hash(tmp_path: Path) -> None:
    cfg = _cfg()
    first = run_stage24_capital_sim(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        base_timeframe="1m",
        operational_timeframe="1h",
        mode="risk_pct",
        initial_equities=[100.0, 1000.0],
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
        docs_dir=tmp_path / "docs",
    )
    second = run_stage24_capital_sim(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        base_timeframe="1m",
        operational_timeframe="1h",
        mode="risk_pct",
        initial_equities=[100.0, 1000.0],
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
        docs_dir=tmp_path / "docs",
    )
    hash_a = str(dict(first["summary"]).get("results_hash", ""))
    hash_b = str(dict(second["summary"]).get("results_hash", ""))
    assert hash_a != ""
    assert hash_a == hash_b
