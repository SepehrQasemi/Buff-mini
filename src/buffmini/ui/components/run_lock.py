"""Single active-run lock management for Stage-5 UI."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from buffmini.constants import RUNS_DIR


LOCK_FILENAME = "_active_run.json"


def lock_path(runs_dir: Path = RUNS_DIR) -> Path:
    """Return lock-file path under runs/."""

    return Path(runs_dir) / LOCK_FILENAME


def is_pid_alive(pid: int) -> bool:
    """Best-effort PID liveness check across platforms."""

    if int(pid) <= 0:
        return False
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False


def ensure_lock_sane(runs_dir: Path = RUNS_DIR) -> None:
    """Clear stale lock file if target PID is no longer alive."""

    path = lock_path(runs_dir)
    if not path.exists():
        return
    payload = _safe_load_json(path)
    pid = int(payload.get("pid", 0))
    status = str(payload.get("status", "running"))
    if status != "running":
        path.unlink(missing_ok=True)
        return
    if not is_pid_alive(pid):
        path.unlink(missing_ok=True)


def get_active_run(runs_dir: Path = RUNS_DIR) -> dict[str, Any] | None:
    """Return active lock payload if still valid."""

    ensure_lock_sane(runs_dir)
    path = lock_path(runs_dir)
    if not path.exists():
        return None
    payload = _safe_load_json(path)
    if not payload:
        return None
    return payload


def acquire_lock(run_id: str, pid: int, command: str, runs_dir: Path = RUNS_DIR) -> None:
    """Acquire single-run lock, failing if an active process exists."""

    ensure_lock_sane(runs_dir)
    active = get_active_run(runs_dir)
    if active is not None:
        raise RuntimeError(f"Active run already in progress: {active.get('run_id')}")

    payload = {
        "run_id": str(run_id),
        "pid": int(pid),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": str(command),
        "status": "running",
    }
    path = lock_path(runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def release_lock(run_id: str, runs_dir: Path = RUNS_DIR) -> None:
    """Release lock if held by run_id (or clear stale lock)."""

    path = lock_path(runs_dir)
    if not path.exists():
        return
    payload = _safe_load_json(path)
    if not payload:
        path.unlink(missing_ok=True)
        return
    if str(payload.get("run_id")) == str(run_id) or not is_pid_alive(int(payload.get("pid", 0))):
        path.unlink(missing_ok=True)


def _safe_load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

