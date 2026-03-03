from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_module() -> object:
    script_path = Path("scripts/run_stage27_rerun_all.py")
    spec = importlib.util.spec_from_file_location("stage27_rerun_all_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage27_rerun_orchestrator_contract(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    def _fake_run_command(cmd: list[str]) -> int:
        text = " ".join(cmd)
        if "run_stage24_audit.py" in text:
            (docs_dir / "stage24_report_summary.json").write_text(
                json.dumps(
                    {
                        "verdict": "IMPROVED",
                        "risk_pct_mode": {
                            "run_id": "r24",
                            "exp_lcb": 0.1,
                            "trade_count": 10,
                            "zero_trade_pct": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
        elif "run_stage25B_edge_program.py" in text and "--mode research" in text:
            (docs_dir / "stage25B_edge_program_summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "r25r",
                        "status": "NO_EDGE_IN_RESEARCH",
                        "metrics": {"exp_lcb_best": 0.0, "trade_count_total": 50, "zero_trade_pct": 20.0},
                    }
                ),
                encoding="utf-8",
            )
        elif "run_stage25B_edge_program.py" in text and "--mode live" in text:
            (docs_dir / "stage25B_edge_program_summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "r25l",
                        "status": "NO_EDGE_IN_LIVE",
                        "metrics": {"exp_lcb_best": -0.1, "trade_count_total": 25, "zero_trade_pct": 40.0},
                    }
                ),
                encoding="utf-8",
            )
        elif "run_stage26.py" in text:
            (docs_dir / "stage26_report_summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "r26",
                        "verdict": "NO_EDGE",
                        "conditional_policy_metrics_live": {
                            "exp_lcb": -0.2,
                            "trade_count": 30,
                            "zero_trade_pct": 35.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
        elif "run_stage27_rolling_discovery.py" in text:
            (docs_dir / "stage27_research_engine_summary.json").write_text(
                json.dumps({"run_id": "r27", "rows": 15}),
                encoding="utf-8",
            )
        return 0

    monkeypatch.setattr(module, "_run_command", _fake_run_command)
    args = argparse.Namespace(
        config=Path("configs/default.yaml"),
        seed=42,
        dry_run=True,
        allow_insufficient_data=True,
        symbols="BTC/USDT,ETH/USDT",
        timeframes="15m,1h",
        data_dir=tmp_path / "data",
        derived_dir=tmp_path / "derived",
        docs_dir=docs_dir,
    )
    payload = module.run_stage27_rerun_all(args)
    assert payload["stage"] == "27.5"
    assert payload["coverage_gate_status"] in {"ALLOW_INSUFFICIENT_DATA", "OK"}
    assert len(payload["stages"]) == 5
    assert (docs_dir / "stage27_rerun_report.md").exists()
    assert (docs_dir / "stage27_rerun_summary.json").exists()
