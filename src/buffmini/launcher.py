"""Stage-5.7 local launcher helpers for one-click app startup."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import venv
import webbrowser
from pathlib import Path
from typing import Any


def find_repo_root(start: Path) -> Path:
    """Find repository root by ascending until pyproject.toml is found."""

    cursor = Path(start).resolve()
    if cursor.is_file():
        cursor = cursor.parent

    for candidate in [cursor, *cursor.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate

    raise FileNotFoundError("Could not locate repository root (pyproject.toml not found).")


def venv_python(venv_dir: Path) -> Path:
    """Return interpreter path inside the local virtual environment."""

    return _venv_python_for_os(venv_dir=venv_dir, os_name=os.name)


def ensure_venv(repo_root: Path, venv_dir: Path) -> None:
    """Create local virtual environment if missing."""

    _ = Path(repo_root)
    py_exe = venv_python(venv_dir)
    if py_exe.exists():
        return

    builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=False, upgrade=False)
    builder.create(str(venv_dir))


def ensure_deps(repo_root: Path, py_exe: Path) -> dict[str, Any]:
    """Install or refresh dependencies in the local virtual environment."""

    pyproject = Path(repo_root) / "pyproject.toml"
    stamp_path = Path(py_exe).parent.parent / ".buffmini_deps_state.json"
    expected_hash = hashlib.sha256(pyproject.read_bytes()).hexdigest()

    state = _safe_json(stamp_path)
    deps_ok = _dependencies_present(py_exe=py_exe)
    if deps_ok and state.get("pyproject_hash") == expected_hash:
        return {"installed": False, "error": None}

    commands = [
        [str(py_exe), "-m", "pip", "install", "--upgrade", "pip"],
        [str(py_exe), "-m", "pip", "install", "-e", "."],
    ]
    for command in commands:
        process = subprocess.run(
            command,
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            check=False,
        )
        if process.returncode != 0:
            output = process.stdout[-4000:] if process.stdout else ""
            return {
                "installed": False,
                "error": f"Dependency install failed: {' '.join(command)}\n{output}",
            }

    stamp_payload = {
        "pyproject_hash": expected_hash,
        "python": str(py_exe),
    }
    _write_json(stamp_path, stamp_payload)
    return {"installed": True, "error": None}


def find_free_port(preferred: int = 8501) -> int:
    """Find first available localhost TCP port, preferring the provided value."""

    if _is_bindable(preferred):
        return int(preferred)

    for candidate in range(int(preferred) + 1, int(preferred) + 101):
        if _is_bindable(candidate):
            return candidate

    raise RuntimeError("Unable to find free Streamlit port in range [preferred, preferred+100].")


def run_streamlit(repo_root: Path, py_exe: Path, port: int) -> subprocess.Popen[str]:
    """Start Streamlit UI process from repo root."""

    command = [
        str(py_exe),
        "-m",
        "streamlit",
        "run",
        "src/buffmini/ui/app.py",
        "--server.port",
        str(int(port)),
        "--server.headless",
        "true",
    ]
    return subprocess.Popen(
        command,
        cwd=str(repo_root),
        shell=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def open_browser(url: str) -> None:
    """Best-effort browser open helper."""

    try:
        webbrowser.open(url, new=2)
    except Exception:
        return


def port_in_use(port: int) -> bool:
    """Return True when localhost port cannot be bound."""

    return not _is_bindable(int(port))


def _is_bindable(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _dependencies_present(py_exe: Path) -> bool:
    check_cmd = [
        str(py_exe),
        "-c",
        "import streamlit,pandas,numpy,yaml,pyarrow,plotly,ccxt; print('ok')",
    ]
    process = subprocess.run(
        check_cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        check=False,
    )
    return process.returncode == 0


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def bootstrap_python() -> list[str]:
    """Return a command prefix to run Python on the local machine."""

    if os.name == "nt":
        py_launcher = subprocess.run(
            ["py", "-3", "-c", "import sys; print(sys.version)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            check=False,
        )
        if py_launcher.returncode == 0:
            return ["py", "-3"]
    return [sys.executable]


def _venv_python_for_os(venv_dir: Path, os_name: str) -> Path:
    """Resolve interpreter path for a provided OS name (test helper)."""

    resolved = Path(venv_dir)
    if str(os_name) == "nt":
        return resolved / "Scripts" / "python.exe"
    return resolved / "bin" / "python"
