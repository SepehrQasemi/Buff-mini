"""Super run autosave unit tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from buffmini.ui.components import run_exec


def test_run_export_to_library_invokes_whitelisted_script(monkeypatch, tmp_path) -> None:
    called: dict[str, object] = {}

    def fake_run(args, cwd, capture_output, text, check, shell):
        called["args"] = args
        called["cwd"] = cwd
        called["capture_output"] = capture_output
        called["text"] = text
        called["check"] = check
        called["shell"] = shell
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps({"strategy_id": "s1", "display_name": "Auto"}),
            stderr="",
        )

    monkeypatch.setattr(run_exec.subprocess, "run", fake_run)

    payload = run_exec.run_export_to_library(
        run_id="pipeline_x",
        display_name="Auto",
        runs_dir=tmp_path / "runs",
        library_dir=tmp_path / "library",
    )

    assert payload["strategy_id"] == "s1"
    args = called["args"]
    assert isinstance(args, list)
    assert "scripts/export_to_library.py" in args
    assert "--run-id" in args
    assert "pipeline_x" in args
    assert called["shell"] is False
