from __future__ import annotations

from buffmini.stage26.data_master_audit import build_master_summary


def test_stage26_9_master_audit_schema_keys() -> None:
    raw_payload = {
        "rows": [
            {"symbol": "BTC/USDT", "coverage_years": 4.1, "gaps_detected": {"count": 1}},
            {"symbol": "ETH/USDT", "coverage_years": 3.9, "gaps_detected": {"count": 2}},
        ]
    }
    canonical_payload = {
        "integrity_pass": True,
        "rows": [
            {"symbol": "BTC/USDT", "timeframe": "1h", "rows": 100},
            {"symbol": "ETH/USDT", "timeframe": "1h", "rows": 120},
        ],
    }
    derived_payload = {"integrity_pass": True, "supported": ["3h", "45m"], "checks": []}
    disk_usage = {"raw": 1.0, "canonical": 2.0, "derived": 3.0, "total": 6.0}

    summary = build_master_summary(
        raw_payload=raw_payload,
        canonical_payload=canonical_payload,
        derived_payload=derived_payload,
        disk_usage=disk_usage,
        raw_exit_code=2,
        canonical_exit_code=0,
    )

    for key in (
        "stage",
        "coverage_years_per_symbol",
        "canonical_candle_counts_per_tf",
        "derived_tf_supported",
        "integrity_pass",
        "gaps_detected",
        "disk_usage_mb",
        "raw_audit_exit_code",
        "canonical_audit_exit_code",
        "derived_sanity",
    ):
        assert key in summary
    assert summary["stage"] == "26.9.4"
    assert summary["coverage_years_per_symbol"]["BTC/USDT"] == 4.1
    assert summary["raw_audit_exit_code"] == 2
