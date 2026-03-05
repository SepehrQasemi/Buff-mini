from __future__ import annotations

import json
from pathlib import Path

import pytest

from buffmini.data.coinapi.client import CoinAPIClient
from buffmini.data.coinapi.usage import CoinAPIUsageLedger, build_usage_summary


def test_usage_ledger_summary_aggregates(tmp_path: Path) -> None:
    ledger = CoinAPIUsageLedger(tmp_path / "usage_ledger.jsonl")
    ledger.append(
        {
            "endpoint_name": "funding_rates",
            "status_code": 200,
            "response_bytes": 120,
            "symbol": "BTC/USDT",
            "header_signals": {"x_credits_remaining": "100"},
            "retry_count": 0,
        }
    )
    ledger.append(
        {
            "endpoint_name": "funding_rates",
            "status_code": 500,
            "response_bytes": 50,
            "symbol": "ETH/USDT",
            "header_signals": {"x_credits_remaining": "99"},
            "retry_count": 1,
        }
    )
    rows = ledger.load_records()
    summary = build_usage_summary(rows)
    assert summary["total_requests"] == 2
    assert summary["total_success"] == 1
    assert summary["total_fail"] == 1
    assert summary["status_code_counts"]["200"] == 1
    assert summary["status_code_counts"]["500"] == 1
    assert summary["per_endpoint"]["funding_rates"]["requests"] == 2
    assert summary["per_symbol"]["BTC/USDT"]["bytes"] == 120


def test_coinapi_client_logs_success_and_failure(tmp_path: Path) -> None:
    ledger_path = tmp_path / "usage.jsonl"
    ledger = CoinAPIUsageLedger(ledger_path)
    calls: list[str] = []

    def ok_transport(url: str, headers: dict[str, str], timeout_sec: int):  # noqa: ANN202
        calls.append(url)
        body = json.dumps({"ok": True}).encode("utf-8")
        return 200, body, {"X-Credits-Remaining": "50"}

    client = CoinAPIClient(
        "test_key_1234",
        base_url="https://rest.coinapi.io",
        sleep_ms=0,
        max_total_requests=3,
        ledger=ledger,
        transport=ok_transport,
    )
    payload, meta = client.request_json(
        "/v1/funding",
        params={"symbol_id": "BINANCE_PERP_BTC_USDT"},
        endpoint_name="funding_rates",
        symbol="BTC/USDT",
    )
    assert payload == {"ok": True}
    assert meta.status_code == 200
    assert len(calls) == 1
    rows = ledger.load_records()
    assert len(rows) == 1
    assert rows[0]["endpoint_name"] == "funding_rates"
    assert rows[0]["status_code"] == 200

    def fail_transport(url: str, headers: dict[str, str], timeout_sec: int):  # noqa: ANN202
        raise TimeoutError("boom")

    client_fail = CoinAPIClient(
        "test_key_1234",
        sleep_ms=0,
        max_total_requests=2,
        max_retries=1,
        ledger=ledger,
        transport=fail_transport,
    )
    with pytest.raises(Exception):
        client_fail.request_json("/v1/funding", params={"symbol_id": "BINANCE_PERP_ETH_USDT"})
    rows2 = ledger.load_records()
    assert len(rows2) == 2
    assert int(rows2[-1]["status_code"]) == 0
    assert "boom" in str(rows2[-1]["error_message"])
