"""Stage-5 run-lock behavior tests."""

from __future__ import annotations

import json

from buffmini.ui.components import run_lock


def test_lock_acquire_and_block_second_acquire(tmp_path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: True)
    run_lock.acquire_lock(run_id="run_a", pid=111, command="python scripts/run_pipeline.py", runs_dir=runs_dir)

    active = run_lock.get_active_run(runs_dir=runs_dir)
    assert active is not None
    assert active["run_id"] == "run_a"

    try:
        run_lock.acquire_lock(run_id="run_b", pid=222, command="python scripts/run_pipeline.py", runs_dir=runs_dir)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_stale_lock_is_cleared(tmp_path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    lock_path = runs_dir / "_active_run.json"
    lock_path.write_text(
        json.dumps(
            {
                "run_id": "stale_run",
                "pid": 999999,
                "started_at": "2026-01-01T00:00:00Z",
                "command": "python scripts/run_pipeline.py",
                "status": "running",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_lock, "is_pid_alive", lambda pid: False)
    run_lock.ensure_lock_sane(runs_dir=runs_dir)
    assert not lock_path.exists()
