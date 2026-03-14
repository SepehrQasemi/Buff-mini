from __future__ import annotations

from types import SimpleNamespace

import pytest

from buffmini.data.coinapi.client import CoinAPIRequestError
from scripts import coinapi_verify


def test_coinapi_verify_missing_key_reports_blocked(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(coinapi_verify, "resolve_coinapi_key_status", lambda **kwargs: (False, "MISSING"))
    monkeypatch.setattr(coinapi_verify, "resolve_coinapi_key", lambda **kwargs: None)
    with pytest.raises(SystemExit, match="1"):
        coinapi_verify.main()
    captured = capsys.readouterr()
    assert "ok=false" in captured.out
    assert "key_source=MISSING" in captured.out


def test_coinapi_verify_success_does_not_print_key(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    sentinel = "FAKEKEY123"
    monkeypatch.setattr(coinapi_verify, "resolve_coinapi_key_status", lambda **kwargs: (True, "ENV"))
    monkeypatch.setattr(coinapi_verify, "resolve_coinapi_key", lambda **kwargs: sentinel)

    class _Client:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            pass

        def request_json(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return {}, SimpleNamespace(status_code=200)

    monkeypatch.setattr(coinapi_verify, "CoinAPIClient", _Client)
    coinapi_verify.main()
    captured = capsys.readouterr()
    assert "ok=true http_status=200 key_source=ENV reason=ok" in captured.out
    assert sentinel not in captured.out


def test_coinapi_verify_401_reports_auth_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(coinapi_verify, "resolve_coinapi_key_status", lambda **kwargs: (True, "SECRETS_TXT"))
    monkeypatch.setattr(coinapi_verify, "resolve_coinapi_key", lambda **kwargs: "KEY_DUMMY")

    class _Client:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            pass

        def request_json(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise CoinAPIRequestError("HTTP Error 401: Unauthorized")

    monkeypatch.setattr(coinapi_verify, "CoinAPIClient", _Client)
    with pytest.raises(SystemExit, match="1"):
        coinapi_verify.main()
    captured = capsys.readouterr()
    assert "ok=false http_status=401 key_source=SECRETS_TXT reason=unauthorized" in captured.out

