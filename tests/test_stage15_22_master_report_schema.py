from __future__ import annotations

from buffmini.alpha_v2.reports import summary_hash
from buffmini.alpha_v2.master import build_master_summary


def test_stage15_22_master_report_schema() -> None:
    summaries = {
        "15": {"run_id": "r15", "status": "PASS", "exp_lcb": 0.1, "trade_count": 10, "max_drawdown": 0.1},
        "16": {"run_id": "r16", "status": "PASS"},
        "17": {"run_id": "r17", "status": "PASS", "exp_lcb": 0.2, "trade_count": 11, "max_drawdown": 0.2},
        "18": {"run_id": "r18", "status": "PASS"},
        "19": {"run_id": "r19", "status": "PASS", "exp_lcb": 0.3, "trade_count": 12, "max_drawdown": 0.2},
        "20": {"run_id": "r20", "status": "FAILED"},
        "21": {"run_id": "r21", "status": "PASS"},
        "22": {"run_id": "r22", "status": "PASS", "exp_lcb": 0.05, "trade_count": 8, "max_drawdown": 0.3},
    }
    payload = build_master_summary(summaries=summaries, seed=42)
    assert payload["stage"] == "15_22"
    assert "ab_milestones" in payload
    assert "final_verdict" in payload
    assert payload["best_stage"] in {"15", "17", "19", "22"}
    # stable hash helper available and deterministic on summary payload.
    h1 = summary_hash(payload)
    h2 = summary_hash(payload)
    assert h1 == h2
