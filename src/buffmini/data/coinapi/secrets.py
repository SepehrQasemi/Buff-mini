"""Secret-resolution helpers for CoinAPI credentials.

Resolution order:
1) environment variable COINAPI_KEY
2) repo-local secrets/coinapi_key.txt
3) repo-local secrets/coinapi_key.json
4) repo-local .env with COINAPI_KEY=...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


ENV_SOURCE: Final[str] = "ENV"
TXT_SOURCE: Final[str] = "SECRETS_TXT"
JSON_SOURCE: Final[str] = "SECRETS_JSON"
DOTENV_SOURCE: Final[str] = "DOTENV"


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


def _read_json_key(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ""
    return _clean_key(payload.get("COINAPI_KEY"))


def _read_dotenv_key(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "COINAPI_KEY":
            continue
        cleaned = value.strip().strip("'").strip('"')
        if cleaned:
            return cleaned
    return ""


def resolve_coinapi_key_with_source(*, repo_root: Path | None = None) -> CoinAPIKeyResolution | None:
    """Resolve CoinAPI key using local source priority without printing secrets."""

    env_key = _clean_key(os.getenv("COINAPI_KEY"))
    if env_key:
        return CoinAPIKeyResolution(source=ENV_SOURCE, key=env_key, path=None)

    root = _repo_root(repo_root)
    txt_path = root / "secrets" / "coinapi_key.txt"
    txt_key = _read_text_key(txt_path)
    if txt_key:
        return CoinAPIKeyResolution(source=TXT_SOURCE, key=txt_key, path=str(txt_path.as_posix()))

    json_path = root / "secrets" / "coinapi_key.json"
    json_key = _read_json_key(json_path)
    if json_key:
        return CoinAPIKeyResolution(source=JSON_SOURCE, key=json_key, path=str(json_path.as_posix()))

    dotenv_path = root / ".env"
    dotenv_key = _read_dotenv_key(dotenv_path)
    if dotenv_key:
        return CoinAPIKeyResolution(source=DOTENV_SOURCE, key=dotenv_key, path=str(dotenv_path.as_posix()))

    return None


def resolve_coinapi_key(*, repo_root: Path | None = None) -> str | None:
    """Resolve CoinAPI key or return None when no source contains a non-empty key."""

    resolved = resolve_coinapi_key_with_source(repo_root=repo_root)
    if resolved is None:
        return None
    return resolved.key

