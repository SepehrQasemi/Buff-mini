from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from buffmini.data.snapshot import build_snapshot_payload, snapshot_metadata_from_config, write_snapshot_file
from buffmini.stage24 import audit as stage24_audit
from buffmini.stage25.edge_program import _build_summary


def test_snapshot_metadata_from_config_reads_frozen_file(tmp_path: Path) -> None:
    snap_path = tmp_path / "data" / "snapshots" / "DATA_FROZEN_v1.json"
    payload = build_snapshot_payload(
        snapshot_id="DATA_FROZEN_v1",
        symbols=[],
        timeframes=[],
    )
    write_snapshot_file(snap_path, payload)
    meta = snapshot_metadata_from_config(
        {
            "data": {
                "snapshot": {
                    "id": "DATA_FROZEN_v1",
                    "path": str(snap_path),
                }
            }
        }
    )
    assert meta["data_snapshot_id"] == "DATA_FROZEN_v1"
    assert meta["data_snapshot_hash"] == payload["snapshot_hash"]
    assert meta["data_snapshot_exists"] is True


def test_stage24_and_stage25_summaries_include_snapshot_fields(tmp_path: Path, monkeypatch) -> None:
    snap_path = tmp_path / "data" / "snapshots" / "DATA_FROZEN_v1.json"
    payload = build_snapshot_payload(
        snapshot_id="DATA_FROZEN_v1",
        symbols=[],
        timeframes=[],
    )
    write_snapshot_file(snap_path, payload)
    cfg = {
        "data": {
            "snapshot": {
                "id": "DATA_FROZEN_v1",
                "path": str(snap_path),
            }
        }
    }

    def _fake_trace(**_: object) -> dict:
        return {"run_id": "trace_run", "trace_dir": tmp_path / "trace", "rows": pd.DataFrame(), "summary": {}}

    def _fake_mode_metrics(_: dict) -> dict:
        return {
            "run_id": "trace_run",
            "seed": 42,
            "config_hash": "cfg",
            "data_hash": "data",
            "resolved_end_ts": "2026-03-03T00:00:00+00:00",
            "trade_count": 1.0,
            "zero_trade_pct": 0.0,
            "invalid_pct": 0.0,
            "walkforward_executed_true_pct": 100.0,
            "mc_trigger_rate": 100.0,
            "invalid_order_pct": 0.0,
            "top_reject_reason": "VALID",
            "top_reject_reasons": [],
            "stage24_valid_count": 1,
            "stage24_invalid_count": 0,
            "shadow_live_summary": {},
        }

    def _fake_capital_sim(**_: object) -> dict:
        return {
            "summary": {
                "run_id": "cap_run",
                "results_hash": "abc",
                "scale_invariance_check": {},
                "rows": [],
            }
        }

    monkeypatch.setattr(stage24_audit, "_run_mode_trace", _fake_trace)
    monkeypatch.setattr(stage24_audit, "_mode_metrics", _fake_mode_metrics)
    monkeypatch.setattr(stage24_audit, "run_stage24_capital_sim", _fake_capital_sim)

    out = stage24_audit.run_stage24_audit(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        base_timeframe="1m",
        operational_timeframe="1h",
        initial_equities=[1000.0],
        runs_root=tmp_path / "runs",
        data_dir=tmp_path / "data",
        derived_dir=tmp_path / "derived",
        docs_dir=tmp_path / "docs",
    )
    stage24_payload = dict(out["payload"])
    assert stage24_payload["data_snapshot_id"] == "DATA_FROZEN_v1"
    assert stage24_payload["data_snapshot_hash"] == payload["snapshot_hash"]

    s25 = _build_summary(
        run_id="r",
        seed=42,
        mode="research",
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        families=["price"],
        composers=["weighted_sum"],
        cost_levels=["realistic"],
        rows_df=pd.DataFrame(),
        trace_refs=[],
        runtime_seconds=0.1,
        snapshot_meta=snapshot_metadata_from_config(cfg),
    )
    assert s25["data_snapshot_id"] == "DATA_FROZEN_v1"
    assert s25["data_snapshot_hash"] == payload["snapshot_hash"]

