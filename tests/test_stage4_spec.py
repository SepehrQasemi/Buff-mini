"""Stage-4 trading spec generation tests."""

from __future__ import annotations

from pathlib import Path

from buffmini.spec.trading_spec import generate_trading_spec


def _cfg() -> dict:
    return {
        "universe": {"symbols": ["BTC/USDT", "ETH/USDT"], "timeframe": "1h"},
        "costs": {"round_trip_cost_pct": 0.1, "slippage_pct": 0.0005, "funding_pct_per_day": 0.0},
        "execution": {
            "mode": "net",
            "per_symbol_netting": True,
            "allow_opposite_signals": False,
            "symbol_scope": "per_symbol",
        },
        "risk": {
            "max_gross_exposure": 5.0,
            "max_net_exposure_per_symbol": 5.0,
            "max_open_positions": 10,
            "sizing": {"mode": "risk_budget", "risk_per_trade_pct": 1.0, "fixed_fraction_pct": 10.0},
            "killswitch": {
                "enabled": True,
                "max_daily_loss_pct": 5.0,
                "max_peak_to_valley_dd_pct": 20.0,
                "max_consecutive_losses": 8,
                "cool_down_bars": 48,
            },
            "reeval": {"cadence": "weekly", "min_new_bars": 168},
        },
        "evaluation": {
            "stage4": {
                "default_method": "equal",
                "default_leverage": 1.0,
            }
        },
    }


def test_trading_spec_contains_mode_leverage_caps_and_killswitch(tmp_path: Path) -> None:
    output = tmp_path / "docs" / "trading_spec.md"
    selected_candidates = [
        {
            "candidate_id": "c1",
            "strategy_family": "TrendPullback",
            "gating": "vol+regime",
            "exit_mode": "fixed_atr",
            "parameters": {"atr_sl_multiplier": 1.5},
            "weight": 0.6,
        },
        {
            "candidate_id": "c2",
            "strategy_family": "DonchianBreakout",
            "gating": "vol",
            "exit_mode": "trailing_atr",
            "parameters": {"atr_tp_multiplier": 3.0},
            "weight": 0.4,
        },
    ]
    stage2_metadata = {"run_id": "stage2_x", "stage1_run_id": "stage1_x"}
    stage3_choice = {"overall_choice": {"status": "OK", "method": "equal", "chosen_leverage": 5.0}}
    paths = generate_trading_spec(
        cfg=_cfg(),
        stage2_metadata=stage2_metadata,
        stage3_3_choice=stage3_choice,
        selected_candidates=selected_candidates,
        output_path=output,
    )

    text = paths["trading_spec_path"].read_text(encoding="utf-8")
    assert "Execution mode" in text
    assert "Selected leverage: `5.0x`" in text
    assert "max_gross_exposure" in text
    assert "Kill-Switch" in text
    assert "c1" in text and "c2" in text
    assert "0.600000" in text and "0.400000" in text
    assert paths["paper_checklist_path"].exists()

