"""Stage-5.7 launcher helper tests (offline)."""

from __future__ import annotations

import socket
from pathlib import Path

from buffmini import launcher


def test_find_repo_root_from_nested_dir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "a" / "b" / "c"
    nested.mkdir(parents=True, exist_ok=True)
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    resolved = launcher.find_repo_root(nested)
    assert resolved == repo


def test_venv_python_path_resolution() -> None:
    venv_dir = Path("example_venv")
    win_path = launcher._venv_python_for_os(venv_dir, "nt")
    assert str(win_path).endswith(str(Path("Scripts") / "python.exe"))

    posix_path = launcher._venv_python_for_os(venv_dir, "posix")
    assert str(posix_path).endswith(str(Path("bin") / "python"))


def test_find_free_port_returns_bindable_port() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    busy_port = int(sock.getsockname()[1])
    sock.listen(1)

    try:
        free_port = launcher.find_free_port(preferred=busy_port)
        assert free_port != busy_port

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", free_port))
        finally:
            probe.close()
    finally:
        sock.close()
