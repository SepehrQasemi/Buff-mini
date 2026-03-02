from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage24.capital_sim import run_stage24_capital_sim


def test_stage24_no_pathological_margin_fail_between_1k_and_10k(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.setdefault("risk", {})["max_gross_exposure"] = 5.0
    cfg.setdefault("evaluation", {}).setdefault("stage23", {})["enabled"] = False

    result = run_stage24_capital_sim(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        base_timeframe="1m",
        operational_timeframe="1h",
        mode="risk_pct",
        initial_equities=[1000.0, 10000.0],
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
        docs_dir=tmp_path / "docs",
    )
    rows = sorted(result["summary"]["rows"], key=lambda item: float(item["initial_equity"]))
    row_1k = rows[0]
    row_10k = rows[1]

    if float(row_1k["trade_count"]) > 0:
        pathological = (
            float(row_10k["trade_count"]) <= 0.0
            and float(row_10k["invalid_order_pct"]) >= 99.999
            and str(row_10k["top_invalid_reason"]) == "MARGIN_FAIL"
        )
        assert pathological is False
        if float(row_10k["trade_count"]) <= 0.0:
            assert str(row_10k["top_invalid_reason"]) in {"POLICY_CAP_HIT", "SIZE_TOO_SMALL", "VALID"}
