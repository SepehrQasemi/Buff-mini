from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research import scope_ladder


def test_stage91_scope_ladder_outputs_rows_and_regime_map(monkeypatch) -> None:
    cfg = load_config(Path(DEFAULT_CONFIG_PATH))

    def fake_campaign(
        *,
        config,
        symbol,
        timeframe,
        families,
        candidate_limit,
        requested_mode,
        auto_pin_resolved_end,
        relax_continuity,
        evaluate_transfer,
    ):
        assert evaluate_transfer is False
        return {
            "candidate_count": 12,
            "promising_count": 3,
            "validated_count": 1,
            "robust_count": 0,
            "blocked_count": 0,
            "dominant_failure_reasons": {"walkforward": 2},
        }

    monkeypatch.setattr(scope_ladder, "evaluate_scope_campaign", fake_campaign)
    monkeypatch.setattr(scope_ladder, "discover_transfer_symbols", lambda config: ["BTC/USDT", "ETH/USDT"])
    ladder = scope_ladder.evaluate_scope_ladder(cfg, feedback={})
    assert ladder["rows"]
    assert set(item["regime"] for item in ladder["regime_scope_map"]) == set(scope_ladder.REGIME_LADDER)
    assert "BTC/USDT" in ladder["tier1_symbols"]
