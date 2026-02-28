"""Stage-9 impact analysis tests."""

from __future__ import annotations

import pandas as pd

from buffmini.analysis.impact_analysis import (
    analyze_symbol_impact,
    bootstrap_median_difference,
    compute_forward_returns,
)


def _frame(rows: int = 220) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = [100 + i * 0.1 for i in range(rows)]
    base = pd.DataFrame(
        {
            "timestamp": ts,
            "close": close,
            "funding_z_30": [0.1 if i % 2 == 0 else -0.1 for i in range(rows)],
            "oi_z_30": [0.2 if i % 3 == 0 else -0.2 for i in range(rows)],
            "funding_extreme_pos": [1 if i % 11 == 0 else 0 for i in range(rows)],
            "funding_extreme_neg": [1 if i % 13 == 0 else 0 for i in range(rows)],
            "crowd_long_risk": [1 if i % 17 == 0 else 0 for i in range(rows)],
            "crowd_short_risk": [1 if i % 19 == 0 else 0 for i in range(rows)],
        }
    )
    return base


def test_bootstrap_median_difference_deterministic() -> None:
    frame = _frame()
    enriched = compute_forward_returns(frame)
    first = bootstrap_median_difference(enriched, "funding_extreme_pos", "forward_return_24h", n_boot=300, seed=42)
    second = bootstrap_median_difference(enriched, "funding_extreme_pos", "forward_return_24h", n_boot=300, seed=42)
    assert first == second
    assert first["ci_low"] <= first["ci_high"]


def test_analyze_symbol_impact_outputs_rows() -> None:
    frame = _frame()
    result = analyze_symbol_impact(frame, symbol="BTC/USDT", seed=42, n_boot=300)
    assert result["symbol"] == "BTC/USDT"
    assert result["rows"]
    assert "best_effect" in result
    assert "corr_funding_z30_vs_fwd24" in result
    assert "corr_oi_z30_vs_fwd24" in result
