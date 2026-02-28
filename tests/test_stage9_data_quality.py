"""Stage-9 data quality summary tests."""

from __future__ import annotations

from buffmini.analysis.impact_analysis import summarize_data_quality


def test_summarize_data_quality_schema() -> None:
    funding_meta = {
        "start_ts": "2025-01-01T00:00:00+00:00",
        "end_ts": "2025-01-10T00:00:00+00:00",
        "row_count": 100,
        "gaps_count": 2,
    }
    oi_meta = {
        "start_ts": "2025-01-01T00:00:00+00:00",
        "end_ts": "2025-01-10T00:00:00+00:00",
        "row_count": 240,
        "gaps_count": 1,
    }

    summary = summarize_data_quality("BTC/USDT", funding_meta, oi_meta)
    assert summary["symbol"] == "BTC/USDT"
    assert summary["funding"]["row_count"] == 100
    assert summary["open_interest"]["row_count"] == 240
    assert summary["funding"]["gaps_count"] == 2
    assert summary["open_interest"]["gaps_count"] == 1
