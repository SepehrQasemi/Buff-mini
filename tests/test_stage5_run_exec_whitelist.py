"""Stage-5 run execution whitelist tests."""

from __future__ import annotations

import pytest

from buffmini.ui.components.run_exec import validate_pipeline_params, validate_whitelisted_script


def test_whitelist_allows_only_known_scripts() -> None:
    assert validate_whitelisted_script("scripts/run_pipeline.py") == "scripts/run_pipeline.py"
    with pytest.raises(ValueError):
        validate_whitelisted_script("scripts/not_allowed.py")


def test_pipeline_param_validation_blocks_bad_symbols_and_invalid_values() -> None:
    valid = validate_pipeline_params(
        {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "1h",
            "window_months": 12,
            "candidate_count": 100,
            "mode": "quick",
            "execution_mode": "net",
            "fees_round_trip_pct": 0.1,
            "seed": 42,
            "run_stage4_simulate": 0,
        }
    )
    assert valid["symbols"] == ["BTC/USDT", "ETH/USDT"]

    with pytest.raises(ValueError):
        validate_pipeline_params(
            {
                "symbols": ["BTC/USDT; rm -rf /"],
                "timeframe": "1h",
                "window_months": 12,
                "candidate_count": 100,
                "mode": "quick",
                "execution_mode": "net",
                "fees_round_trip_pct": 0.1,
                "seed": 42,
                "run_stage4_simulate": 0,
            }
        )

    with pytest.raises(ValueError):
        validate_pipeline_params(
            {
                "symbols": ["BTC/USDT"],
                "timeframe": "4h",
                "window_months": 12,
                "candidate_count": 100,
                "mode": "quick",
                "execution_mode": "net",
                "fees_round_trip_pct": 0.1,
                "seed": 42,
                "run_stage4_simulate": 0,
            }
        )
