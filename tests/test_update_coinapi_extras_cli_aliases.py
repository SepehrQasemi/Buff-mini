from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from buffmini.config import load_config
from scripts import update_coinapi_extras


def _write_cfg(tmp_path: Path, *, enabled: bool) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    base = load_config(repo_root / "configs" / "default.yaml")
    base["coinapi"]["enabled"] = bool(enabled)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return cfg_path


def test_cli_alias_parsing_for_download_years_and_budget() -> None:
    args = update_coinapi_extras.parse_args(
        [
            "--download",
            "--years",
            "4",
            "--budget-requests",
            "1500",
            "--endpoints",
            "funding,oi",
            "--symbols",
            "BTC/USDT,ETH/USDT",
        ]
    )
    assert args.download is True
    assert args.years == 4
    assert args.budget_requests == 1500
    endpoints = update_coinapi_extras.normalize_endpoints(update_coinapi_extras._split_csv(args.endpoints))
    assert endpoints == ["funding_rates", "open_interest"]


def test_cli_unknown_endpoint_lists_allowed_names() -> None:
    with pytest.raises(SystemExit, match="Allowed endpoint names/aliases"):
        update_coinapi_extras.normalize_endpoints(["funding_rates", "unknown_ep"])


def test_dry_run_plan_exits_ok_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _write_cfg(tmp_path, enabled=True)
    monkeypatch.chdir(tmp_path)

    class _NoNetworkClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003, D401
            raise AssertionError("CoinAPIClient must not be instantiated in dry-run plan mode")

    monkeypatch.setattr(update_coinapi_extras, "CoinAPIClient", _NoNetworkClient)
    argv = [
        "update_coinapi_extras.py",
        "--config",
        str(cfg),
        "--plan",
        "--symbols",
        "BTC/USDT,ETH/USDT",
        "--endpoints",
        "funding,oi",
        "--last-days",
        "14",
        "--increment-days",
        "7",
        "--max-requests",
        "200",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    update_coinapi_extras.main()
    captured = capsys.readouterr()
    assert "plan_path:" in captured.out


def test_missing_key_raises_clean_error_without_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_cfg(tmp_path, enabled=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COINAPI_KEY", raising=False)
    argv = [
        "update_coinapi_extras.py",
        "--config",
        str(cfg),
        "--download",
        "--symbols",
        "BTC/USDT",
        "--endpoints",
        "funding",
        "--last-days",
        "3",
        "--max-requests",
        "10",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="COINAPI_KEY missing; use secrets/coinapi_key.txt"):
        update_coinapi_extras.main()


def test_download_refuses_when_plan_exceeds_budget_and_writes_usage_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_cfg(tmp_path, enabled=True)
    monkeypatch.chdir(tmp_path)
    argv = [
        "update_coinapi_extras.py",
        "--config",
        str(cfg),
        "--download",
        "--seed",
        "42",
        "--symbols",
        "BTC/USDT,ETH/USDT",
        "--endpoints",
        "funding,oi",
        "--last-days",
        "365",
        "--increment-days",
        "1",
        "--max-requests",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="Planned request count exceeds --max-requests"):
        update_coinapi_extras.main()

    usage_doc = tmp_path / "docs" / "stage35_7_coinapi_usage.json"
    assert usage_doc.exists()
    payload = yaml.safe_load(usage_doc.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    latest = payload.get("latest", {})
    assert int(latest.get("total_requests_planned", 0)) > int(latest.get("total_requests_selected", 0))
    assert "status_code_counts" in latest
    assert "retry_counts" in latest
