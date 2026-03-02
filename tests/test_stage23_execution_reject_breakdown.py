from __future__ import annotations

import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.forensics.signal_flow import run_signal_flow_trace
from buffmini.stage23.rejects import EXECUTION_REJECT_REASONS, RejectBreakdown, normalize_reject_reason


def test_reject_reason_normalization_known_set() -> None:
    assert normalize_reject_reason("SIZE_ZERO") == "SIZE_ZERO"
    assert normalize_reject_reason("size_zero") == "SIZE_ZERO"
    assert normalize_reject_reason("unknown_reason") == "UNKNOWN"
    assert normalize_reject_reason(None) == "UNKNOWN"


def test_reject_breakdown_logical_totals() -> None:
    breakdown = RejectBreakdown()
    breakdown.register_attempt(10)
    breakdown.register_accept(7)
    breakdown.register_reject("NO_FILL", 2)
    payload = breakdown.to_payload()
    assert payload["total_orders_attempted"] == 10
    assert payload["total_orders_accepted"] == 7
    assert payload["total_orders_rejected"] == 3
    assert int(sum(payload["reject_reason_counts"].values())) == 3
    assert set(payload["reject_reason_counts"]).issuperset(set(EXECUTION_REJECT_REASONS))


def test_execution_reject_breakdown_artifact_exists(tmp_path: Path) -> None:
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
        max_combos=5,
        dry_run=True,
        runs_root=tmp_path,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
    )
    path = result["trace_dir"] / "execution_reject_breakdown.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    attempted = int(payload["total_orders_attempted"])
    accepted = int(payload["total_orders_accepted"])
    rejected = int(payload["total_orders_rejected"])
    assert attempted >= accepted >= 0
    assert attempted == accepted + rejected

