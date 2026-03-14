"""Operator helper for CoinAPI key status and local secret rotation."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path

from buffmini.data.coinapi.secrets import resolve_coinapi_key_with_source


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoinAPI key helper (no secret output)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--status", action="store_true", help="Print OK/MISSING and source name only")
    action.add_argument("--wipe-old", action="store_true", help="Delete legacy local secret files")
    action.add_argument("--write", action="store_true", help="Prompt securely and write .secrets/coinapi_key.txt")
    return parser.parse_args()


def _status(repo_root: Path) -> int:
    resolved = resolve_coinapi_key_with_source(repo_root=repo_root)
    if resolved is None:
        print("MISSING")
        return 1
    print(f"OK source={resolved.source}")
    return 0


def _wipe_old(repo_root: Path) -> int:
    removed: list[str] = []
    for rel in (
        ".secrets/coinapi_key.txt",
        ".secrets/coinapi_key",
        "secrets/coinapi_key.txt",
        "secrets/coinapi_key.json",
    ):
        path = repo_root / rel
        if path.exists() and path.is_file():
            path.unlink()
            removed.append(rel)
    if removed:
        print(f"REMOVED {len(removed)} file(s)")
    else:
        print("NOTHING_TO_REMOVE")
    return 0


def _write(repo_root: Path) -> int:
    try:
        secret = str(getpass.getpass("Enter COINAPI_KEY (input hidden): ")).strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("Unable to read key interactively; create .secrets/coinapi_key.txt manually.")
    if not secret:
        raise SystemExit("Empty key. Create .secrets/coinapi_key.txt manually or rerun with a non-empty key.")
    target = repo_root / ".secrets" / "coinapi_key.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(secret + "\n", encoding="utf-8")
    print("WROTE .secrets/coinapi_key.txt")
    return 0


def main() -> None:
    args = parse_args()
    root = _repo_root()
    if args.status:
        raise SystemExit(_status(root))
    if args.wipe_old:
        raise SystemExit(_wipe_old(root))
    if args.write:
        raise SystemExit(_write(root))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
