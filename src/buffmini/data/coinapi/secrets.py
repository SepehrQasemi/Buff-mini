"""Secret-resolution helpers for CoinAPI credentials.

Resolution order:
1) environment variable COINAPI_KEY
2) repo-local .secrets/coinapi_key.txt
3) repo-local .secrets/coinapi_key
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


ENV_SOURCE: Final[str] = "ENV"
SECRETS_TXT_SOURCE: Final[str] = "SECRETS_TXT"
SECRETS_FILE_SOURCE: Final[str] = "SECRETS_FILE"
MISSING_SOURCE: Final[str] = "MISSING"


@dataclass(frozen=True, slots=True)
class CoinAPIKeyResolution:
    source: str
    key: str
    path: str | None = None


def _clean_key(value: str | None) -> str:
    return str(value or "").strip()


def _repo_root(repo_root: Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    return Path.cwd().resolve()


def _read_text_key(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    return text.splitlines()[0].strip()


def resolve_coinapi_key_with_source(*, repo_root: Path | None = None) -> CoinAPIKeyResolution | None:
    """Resolve CoinAPI key using source priority without printing/logging key content."""

    env_key = _clean_key(os.getenv("COINAPI_KEY"))
    if env_key:
        return CoinAPIKeyResolution(source=ENV_SOURCE, key=env_key, path=None)

    root = _repo_root(repo_root)
    for secrets_dir in (".secrets", "secrets"):
        txt_path = root / secrets_dir / "coinapi_key.txt"
        txt_key = _read_text_key(txt_path)
        if txt_key:
            return CoinAPIKeyResolution(source=SECRETS_TXT_SOURCE, key=txt_key, path=txt_path.as_posix())

        file_path = root / secrets_dir / "coinapi_key"
        file_key = _read_text_key(file_path)
        if file_key:
            return CoinAPIKeyResolution(source=SECRETS_FILE_SOURCE, key=file_key, path=file_path.as_posix())

    return None


def resolve_coinapi_key(*, repo_root: Path | None = None) -> str | None:
    """Resolve CoinAPI key text or return None when unavailable."""

    resolved = resolve_coinapi_key_with_source(repo_root=repo_root)
    if resolved is None:
        return None
    return resolved.key


def resolve_coinapi_key_presence(*, repo_root: Path | None = None) -> tuple[bool, str]:
    """Return key presence and source label without exposing key material."""

    resolved = resolve_coinapi_key_with_source(repo_root=repo_root)
    if resolved is None:
        return False, MISSING_SOURCE
    return True, str(resolved.source)
