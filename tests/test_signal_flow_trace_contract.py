from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.forensics.signal_flow import parse_stage_arg, run_signal_flow_trace, summarize_trace


def test_parse_stage_arg_handles_range() -> None:
    out = parse_stage_arg("15..22")
    assert out == ["15", "16", "17", "18", "19", "20", "21", "22"]


def test_summarize_trace_contract_keys() -> None:
    rows = pd.DataFrame(
        [
            {
                "stage": "15",
                "mode": "v2",
                "timeframe": "1h",
                "family": "price",
                "raw_signal_count": 10,
                "after_context_count": 8,
                "after_confirm_count": 8,
                "after_riskgate_count": 7,
                "orders_sent_count": 7,
                "trades_executed_count": 5,
                "top_reject_reason": "VALID",
                "walkforward_executed_true": 1,
                "MC_triggered": 1,
            },
            {
                "stage": "17",
                "mode": "v2",
                "timeframe": "2h",
                "family": "flow",
                "raw_signal_count": 12,
                "after_context_count": 0,
                "after_confirm_count": 0,
                "after_riskgate_count": 0,
                "orders_sent_count": 0,
                "trades_executed_count": 0,
                "top_reject_reason": "CONTEXT_REJECT",
                "walkforward_executed_true": 0,
                "MC_triggered": 0,
            },
        ]
    )
    reject = pd.DataFrame([{"reason": "VALID"}, {"reason": "CONTEXT_REJECT"}])
    out = summarize_trace(rows_df=rows, reject_reasons=reject)
    for key in (
        "rows_count",
        "top_bottlenecks",
        "by_stage",
        "by_timeframe",
        "by_family",
        "stage_pass_fail",
        "zero_trade_pct",
        "invalid_pct",
    ):
        assert key in out


def test_run_signal_flow_trace_contract(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    result = run_signal_flow_trace(
        config=cfg,
        seed=42,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        mode="classic",
        stages=["classic"],
        families=["price"],
        composers=["none"],
        max_combos=10,
        dry_run=True,
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
    )
    assert "run_id" in result
    assert (result["trace_dir"] / "signal_flow_trace.csv").exists()
    assert (result["trace_dir"] / "signal_flow_trace.json").exists()
    assert (result["trace_dir"] / "reject_reasons.csv").exists()
    assert (result["trace_dir"] / "trace_summary.json").exists()
