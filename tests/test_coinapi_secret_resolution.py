from __future__ import annotations

from pathlib import Path

from buffmini.data.coinapi.secrets import (
    ENV_SOURCE,
    MISSING_SOURCE,
    SECRETS_FILE_SOURCE,
    SECRETS_TXT_SOURCE,
    resolve_coinapi_key,
    resolve_coinapi_key_presence,
    resolve_coinapi_key_with_source,
)


def test_resolve_coinapi_key_prefers_environment(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("COINAPI_KEY", "ENV_KEY_123")
    resolved = resolve_coinapi_key_with_source(repo_root=Path.cwd())
    assert resolved is not None
    assert resolved.source == ENV_SOURCE
    assert resolved.key == "ENV_KEY_123"


def test_resolve_coinapi_key_uses_secrets_txt_when_env_missing(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("COINAPI_KEY", raising=False)
    secrets_dir = tmp_path / ".secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    (secrets_dir / "coinapi_key.txt").write_text("TXT_KEY_456\n", encoding="utf-8")
    resolved = resolve_coinapi_key_with_source(repo_root=tmp_path)
    assert resolved is not None
    assert resolved.source == SECRETS_TXT_SOURCE
    assert resolved.key == "TXT_KEY_456"
    assert resolve_coinapi_key(repo_root=tmp_path) == "TXT_KEY_456"


def test_resolve_coinapi_key_uses_secrets_file_after_txt(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("COINAPI_KEY", raising=False)
    secrets_dir = tmp_path / ".secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    (secrets_dir / "coinapi_key").write_text("FILE_KEY_789\n", encoding="utf-8")
    resolved = resolve_coinapi_key_with_source(repo_root=tmp_path)
    assert resolved is not None
    assert resolved.source == SECRETS_FILE_SOURCE
    assert resolved.key == "FILE_KEY_789"
    present, source = resolve_coinapi_key_presence(repo_root=tmp_path)
    assert present is True
    assert source == SECRETS_FILE_SOURCE


def test_resolve_coinapi_key_supports_legacy_secrets_dir(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("COINAPI_KEY", raising=False)
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    (secrets_dir / "coinapi_key.txt").write_text("LEGACY_KEY_001\n", encoding="utf-8")
    resolved = resolve_coinapi_key_with_source(repo_root=tmp_path)
    assert resolved is not None
    assert resolved.source == SECRETS_TXT_SOURCE
    assert resolved.key == "LEGACY_KEY_001"


def test_resolver_does_not_emit_secret_to_stdout_stderr(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    sentinel = "FAKEKEY123"
    monkeypatch.setenv("COINAPI_KEY", sentinel)
    resolved = resolve_coinapi_key_with_source(repo_root=Path.cwd())
    assert resolved is not None
    assert resolved.key == sentinel
    captured = capsys.readouterr()
    assert sentinel not in captured.out
    assert sentinel not in captured.err


def test_resolve_coinapi_key_presence_reports_missing(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("COINAPI_KEY", raising=False)
    present, source = resolve_coinapi_key_presence(repo_root=tmp_path)
    assert present is False
    assert source == MISSING_SOURCE
