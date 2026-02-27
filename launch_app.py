"""Stage-5.7 one-click local launcher for Buff-mini Streamlit UI."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from buffmini.launcher import (
    ensure_deps,
    ensure_venv,
    find_free_port,
    find_repo_root,
    open_browser,
    port_in_use,
    run_streamlit,
    venv_python,
)


def main() -> int:
    try:
        repo_root = find_repo_root(Path(__file__).resolve().parent)
        venv_dir = repo_root / ".venv"
        ensure_venv(repo_root=repo_root, venv_dir=venv_dir)
        py_exe = venv_python(venv_dir)
        deps_result = ensure_deps(repo_root=repo_root, py_exe=py_exe)
        if deps_result.get("error"):
            print("[ERROR] Dependency setup failed.")
            print(deps_result["error"])
            print("Try running again after fixing Python/pip connectivity.")
            return 1

        preferred_port = 8501
        already_running = port_in_use(preferred_port)
        port = find_free_port(preferred=preferred_port)
        url = f"http://localhost:{port}"
        running_url = f"http://localhost:{preferred_port}"

        print("=" * 56)
        print("Buff-mini Stage-5.7 One-Click Launcher")
        print("=" * 56)
        print(f"Repo root: {repo_root}")
        print(f"Virtual env: {venv_dir}")
        print(f"Dependency install executed: {bool(deps_result.get('installed', False))}")
        if already_running and port != preferred_port:
            print(f"Detected service on {running_url}. It may be an existing Buff-mini UI instance.")
            print(f"Launching a new instance on: {url}")
        else:
            print(f"Starting UI at: {url}")

        process = run_streamlit(repo_root=repo_root, py_exe=py_exe, port=port)
        time.sleep(2)
        open_browser(url)
        print("Browser opened. Press Ctrl+C in this window to stop Streamlit.")

        while True:
            line = process.stdout.readline() if process.stdout is not None else ""
            if line:
                print(line.rstrip())
            if process.poll() is not None:
                return int(process.returncode or 0)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping launcher...")
        return 0
    except Exception as exc:
        print("[ERROR] Launcher failed.")
        print(str(exc))
        print("Troubleshooting: ensure Python 3.11+ is installed, then retry.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
