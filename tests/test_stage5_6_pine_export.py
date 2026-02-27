"""Tests for Stage-5.6 Pine script export."""

from __future__ import annotations

import json
from pathlib import Path

from buffmini.spec.pine_export import PineComponent, PineContext, export_pine_scripts, render_component_pine


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_export_pine_creates_expected_files_and_is_deterministic(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_id = "pipeline_x"
    stage1_run = "stage1_x"
    stage2_run = "stage2_x"

    candidate_payload = {
        "candidate_id": "cand_a",
        "strategy_family": "TrendPullback",
        "gating": "vol+regime",
        "exit_mode": "fixed_atr",
        "parameters": {
            "channel_period": 55,
            "ema_fast": 50,
            "ema_slow": 200,
            "rsi_long_entry": 35,
            "rsi_short_entry": 65,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "atr_sl_multiplier": 1.5,
            "atr_tp_multiplier": 3.0,
            "trailing_atr_k": 1.5,
            "max_holding_bars": 24,
            "regime_gate_long": True,
            "regime_gate_short": True,
        },
    }

    _write_json(
        runs_dir / run_id / "pipeline_summary.json",
        {
            "run_id": run_id,
            "status": "success",
            "stage2_run_id": stage2_run,
            "chosen_method": "equal",
            "chosen_leverage": 5.0,
            "config_hash": "cfg-hash",
            "data_hash": "data-hash",
        },
    )
    _write_json(
        runs_dir / run_id / "ui_bundle" / "summary_ui.json",
        {
            "run_id": run_id,
            "timeframe": "1h",
            "chosen_method": "equal",
            "chosen_leverage": 5.0,
            "execution_mode": "net",
            "seed": 42,
            "selected_components": [
                {
                    "candidate_id": "cand_a",
                    "parameters": {
                        "channel_period": 55,
                        "ema_fast": 50,
                        "ema_slow": 200,
                        "signal_ema": 20,
                        "rsi_period": 14,
                        "rsi_long_entry": 35.0,
                        "rsi_short_entry": 65.0,
                        "bollinger_period": 20,
                        "bollinger_std": 2.0,
                        "atr_sl_multiplier": 1.5,
                        "atr_tp_multiplier": 3.0,
                        "trailing_atr_k": 1.5,
                        "max_holding_bars": 24,
                        "regime_gate_long": True,
                        "regime_gate_short": True,
                    },
                }
            ],
        },
    )
    _write_json(
        runs_dir / stage2_run / "portfolio_summary.json",
        {
            "run_id": stage2_run,
            "stage1_run_id": stage1_run,
            "round_trip_cost_pct": 0.1,
            "slippage_pct": 0.0005,
            "portfolio_methods": {
                "equal": {
                    "weights": {"cand_a": 1.0},
                    "selected_candidates": ["cand_a"],
                }
            },
        },
    )
    (runs_dir / stage1_run / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (runs_dir / stage1_run / "config.yaml").write_text(
        "universe:\n  symbols: [BTC/USDT]\n  timeframe: 1h\nsearch:\n  seed: 42\n",
        encoding="utf-8",
    )
    _write_json(runs_dir / stage1_run / "candidates" / "strategy_01_cand_a.json", candidate_payload)

    first = export_pine_scripts(run_id=run_id, runs_dir=runs_dir)
    second = export_pine_scripts(run_id=run_id, runs_dir=runs_dir)

    export_dir = runs_dir / run_id / "exports" / "pine"
    assert (export_dir / "cand_a.pine.txt").exists()
    assert (export_dir / "portfolio_template.pine.txt").exists()
    assert (export_dir / "index.json").exists()
    text = (export_dir / "cand_a.pine.txt").read_text(encoding="utf-8")
    assert "//@version=5" in text
    assert "strategy(" in text
    assert "input." in text
    assert f"run_id: {run_id}" in text
    assert "lookahead_on" not in text.lower()
    assert first["deterministic_export"] is True
    assert first["validation"]["all_files_valid"] is True
    assert first["files"] == second["files"]


def test_family_mapping_emits_expected_entry_clauses() -> None:
    base_ctx = PineContext(
        run_id="r",
        run_dir=Path("."),
        stage1_run_id="s1",
        stage2_run_id="s2",
        stage3_3_run_id=None,
        chosen_method="equal",
        chosen_leverage=1.0,
        timeframe="1h",
        execution_mode="net",
        config_hash="c",
        data_hash="d",
        seed=42,
        round_trip_cost_pct=0.1,
        slippage_pct=0.0005,
        summary_ui={},
        components=[],
    )
    families = {
        "DonchianBreakout": "close > donchianHigh",
        "RSIMeanReversion": "rsiVal < rsiLongEntry",
        "TrendPullback": "emaFastVal > emaSlowVal and close > signalEmaVal",
        "BollingerMeanReversion": "close < bbLower and rsiVal < rsiLongEntry",
        "RangeBreakoutTrendFilter": "close > donchianHigh and emaFastVal > emaSlowVal",
    }
    params = {
        "channel_period": 20,
        "ema_fast": 50,
        "ema_slow": 200,
        "signal_ema": 20,
        "rsi_period": 14,
        "rsi_long_entry": 30.0,
        "rsi_short_entry": 70.0,
        "bollinger_period": 20,
        "bollinger_std": 2.0,
        "atr_sl_multiplier": 1.5,
        "atr_tp_multiplier": 3.0,
        "trailing_atr_k": 1.5,
        "max_holding_bars": 24,
        "regime_gate_long": True,
        "regime_gate_short": True,
    }
    for family, needle in families.items():
        component = PineComponent(
            candidate_id="cid",
            strategy_family=family,
            strategy_name=family,
            gating_mode="none",
            exit_mode="fixed_atr",
            parameters=dict(params),
            weight=1.0,
        )
        script = render_component_pine(base_ctx, component)
        assert needle in script
