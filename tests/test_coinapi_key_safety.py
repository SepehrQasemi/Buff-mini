from __future__ import annotations

import json
from pathlib import Path

from buffmini.data.coinapi.client import CoinAPIClient
from buffmini.data.coinapi.usage import CoinAPIUsageLedger


def test_coinapi_key_never_written_to_ledger(tmp_path: Path) -> None:
    secret = "COINAPI_SECRET_ABC1234"
    ledger_path = tmp_path / "usage.jsonl"
    ledger = CoinAPIUsageLedger(ledger_path)

    def transport(url: str, headers: dict[str, str], timeout_sec: int):  # noqa: ANN202
        assert headers.get("X-CoinAPI-Key") == secret
        return 200, json.dumps({"rows": []}).encode("utf-8"), {"X-Credits-Remaining": "10"}

    client = CoinAPIClient(secret, sleep_ms=0, max_total_requests=1, ledger=ledger, transport=transport)
    client.request_json("/v1/test", params={"symbol_id": "BINANCE_PERP_BTC_USDT"}, endpoint_name="funding_rates")

    text = ledger_path.read_text(encoding="utf-8")
    assert secret not in text
    assert secret[-4:] in text
    assert "***" in text

