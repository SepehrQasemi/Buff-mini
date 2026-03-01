from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.data.features import calculate_features
from buffmini.signals.families.flow import FlowLiquidityFamily
from buffmini.stage13.evaluate import run_stage13
from buffmini.validation.leakage_harness import synthetic_ohlcv


def test_flow_family_handles_missing_overlays_without_crash() -> None:
    raw = synthetic_ohlcv(rows=420, seed=7)
    frame = calculate_features(raw)
    fam = FlowLiquidityFamily(params={"entry_threshold": 0.3})
    ctx = type("Ctx", (), {"symbol": "BTC/USDT", "timeframe": "1h", "seed": 42, "config": {}, "params": {}})()
    scores = fam.compute_scores(frame, ctx)
    out = fam.propose_entries(scores, frame, ctx)
    assert len(out) == len(frame)
    assert out["signal"].isin([-1, 0, 1]).all()


def test_flow_disabled_non_corruption_for_price_only(tmp_path: Path) -> None:
    cfg = deepcopy(load_config(DEFAULT_CONFIG_PATH))
    cfg["evaluation"]["stage13"]["enabled"] = True
    cfg["evaluation"]["stage13"]["families"]["enabled"] = ["price"]
    a = run_stage13(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframe="1h",
        families=["price"],
        composer_mode="none",
        runs_root=tmp_path / "runs_a",
        docs_dir=tmp_path / "docs_a",
        stage_tag="13.4",
        report_name="stage13_4_flow_family_a",
        write_docs=False,
    )
    cfg["evaluation"]["stage13"]["families"]["enabled"] = ["price", "flow"]
    b = run_stage13(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframe="1h",
        families=["price"],
        composer_mode="none",
        runs_root=tmp_path / "runs_b",
        docs_dir=tmp_path / "docs_b",
        stage_tag="13.4",
        report_name="stage13_4_flow_family_b",
        write_docs=False,
    )
    a_rows = a["rows"].loc[a["rows"]["family"] == "price", ["trade_count", "exp_lcb", "PF"]].reset_index(drop=True)
    b_rows = b["rows"].loc[b["rows"]["family"] == "price", ["trade_count", "exp_lcb", "PF"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a_rows, b_rows, check_exact=False, atol=1e-12, rtol=0.0)

