from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.stage37.activation import compute_reject_chain_metrics
from buffmini.stage38.audit import build_lineage_table_from_stage28


def _write_stage28_fixture(
    root: Path,
    *,
    trace_rows: list[dict[str, object]],
    live_trade_count: float = 0.0,
) -> Path:
    stage28_dir = root / "stage28"
    stage28_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace_rows).to_csv(stage28_dir / "policy_trace.csv", index=False)
    pd.DataFrame(columns=["timestamp", "context", "reason"]).to_csv(stage28_dir / "shadow_live_rejects.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "candidate": "Dummy",
                "family": "flow",
                "context": "RANGE",
                "exp_lcb": 0.0,
            }
        ]
    ).to_csv(stage28_dir / "finalists_stageC.csv", index=False)
    summary = {
        "run_id": "fixture_stage28",
        "policy_metrics": {"live": {"trade_count": float(live_trade_count)}},
    }
    (stage28_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return stage28_dir


def test_stage38_nan_active_candidates_not_counted_as_raw_signal() -> None:
    trace = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="1h", tz="UTC"),
            "context": ["RANGE", "RANGE", "TREND", "TREND"],
            "net_score": [0.0, 0.0, 0.0, 0.0],
            "active_candidates": [np.nan, np.nan, np.nan, np.nan],
            "final_signal": [0, 0, 0, 0],
        }
    )
    out = compute_reject_chain_metrics(
        trace_df=trace,
        shadow_df=pd.DataFrame(columns=["timestamp", "context", "reason"]),
        finalists_df=pd.DataFrame(columns=["candidate_id", "family", "context", "exp_lcb"]),
        threshold=0.0,
        quality_floor=-0.02,
        final_trade_count=0.0,
    )
    overall = dict(out.get("overall", {}))
    assert int(overall.get("raw_signal_count", -1)) == 0
    assert int(overall.get("composer_signal_count", -1)) == 0
    assert float(overall.get("final_trade_count", -1.0)) == 0.0


def test_stage38_lineage_reports_composer_collapse_reason(tmp_path: Path) -> None:
    stage28_dir = _write_stage28_fixture(
        tmp_path,
        trace_rows=[
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "context": "RANGE",
                "net_score": 0.0,
                "active_candidates": "c1",
                "final_signal": 0,
            },
            {
                "timestamp": "2026-01-01T01:00:00Z",
                "context": "RANGE",
                "net_score": 0.0,
                "active_candidates": "c1",
                "final_signal": 0,
            },
        ],
        live_trade_count=0.0,
    )
    table = build_lineage_table_from_stage28(stage28_dir=stage28_dir, threshold=0.0, quality_floor=-0.02)
    assert int(table["raw_signal_count"]) == 2
    assert int(table["composer_signal_count"]) == 0
    assert int(table["engine_raw_signal_count"]) == 0
    assert str(table["collapse_reason"]) == "composer_netting_zero_signal"
    assert bool(table["contradiction_fixed"]) is True

