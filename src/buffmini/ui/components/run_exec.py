"""Safe subprocess execution helpers for Stage-5 UI."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import json
from pathlib import Path
from typing import Any

from buffmini.constants import PROJECT_ROOT, RUNS_DIR
from buffmini.ui.components.run_lock import acquire_lock, get_active_run
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


WHITELISTED_SCRIPTS = {
    "scripts/run_pipeline.py",
    "scripts/run_stage4_spec.py",
    "scripts/run_stage4_simulate.py",
    "scripts/export_to_library.py",
}

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")


def validate_whitelisted_script(script_relpath: str) -> str:
    """Validate script path against Stage-5 execution allow-list."""

    normalized = str(Path(script_relpath).as_posix())
    if normalized not in WHITELISTED_SCRIPTS:
        raise ValueError(f"Script not allowed: {script_relpath}")
    return normalized


def validate_pipeline_params(params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize Strategy Lab parameters for pipeline run."""

    normalized: dict[str, Any] = {}
    symbols = params.get("symbols", [])
    if isinstance(symbols, str):
        symbols = [item.strip() for item in symbols.split(",") if item.strip()]
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("symbols must be a non-empty list")
    normalized_symbols = [str(item).strip() for item in symbols if str(item).strip()]
    if not normalized_symbols:
        raise ValueError("symbols must be non-empty")
    for symbol in normalized_symbols:
        if _SYMBOL_PATTERN.fullmatch(symbol) is None:
            raise ValueError(f"Invalid symbol format: {symbol}")
    normalized["symbols"] = normalized_symbols

    timeframe = str(params.get("timeframe", "1h"))
    if timeframe != "1h":
        raise ValueError("timeframe must be 1h")
    normalized["timeframe"] = timeframe

    window_months = int(params.get("window_months", 36))
    if window_months not in {3, 6, 12, 36}:
        raise ValueError("window_months must be one of: 3,6,12,36")
    normalized["window_months"] = window_months

    mode = str(params.get("mode", "quick")).lower()
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be quick or full")
    normalized["mode"] = mode

    execution_mode = str(params.get("execution_mode", "net")).lower()
    if execution_mode == "auto":
        execution_mode = "net"
    if execution_mode not in {"net", "hedge", "isolated"}:
        raise ValueError("execution_mode must be net|hedge|isolated|auto")
    normalized["execution_mode"] = execution_mode

    candidate_count = int(params.get("candidate_count", 0))
    if candidate_count < 1:
        raise ValueError("candidate_count must be >= 1")
    normalized["candidate_count"] = candidate_count

    fees_round_trip_pct = float(params.get("fees_round_trip_pct", 0.1))
    if not 0 <= fees_round_trip_pct <= 100:
        raise ValueError("fees_round_trip_pct must be between 0 and 100")
    normalized["fees_round_trip_pct"] = fees_round_trip_pct

    seed = int(params.get("seed", 42))
    normalized["seed"] = seed

    run_stage4_simulate = int(params.get("run_stage4_simulate", 0))
    if run_stage4_simulate not in {0, 1}:
        raise ValueError("run_stage4_simulate must be 0 or 1")
    normalized["run_stage4_simulate"] = run_stage4_simulate
    return normalized


def start_pipeline(params: dict[str, Any], runs_dir: Path = RUNS_DIR) -> tuple[str, int]:
    """Start Stage-5 pipeline subprocess and register active-run lock."""

    validated = validate_pipeline_params(params)
    active = get_active_run(runs_dir)
    if active is not None:
        raise RuntimeError(f"Active run exists: {active.get('run_id')}")

    signature = stable_hash(validated, length=10)
    run_id = f"{utc_now_compact()}_{signature}_stage5_pipeline"
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "ui_stdout.log"
    stderr_path = run_dir / "ui_stderr.log"

    script = validate_whitelisted_script("scripts/run_pipeline.py")
    cmd = [
        sys.executable,
        script,
        "--run-id",
        run_id,
        "--symbols",
        ",".join(validated["symbols"]),
        "--timeframe",
        validated["timeframe"],
        "--window-months",
        str(validated["window_months"]),
        "--candidate-count",
        str(validated["candidate_count"]),
        "--mode",
        validated["mode"],
        "--execution-mode",
        validated["execution_mode"],
        "--fees-round-trip-pct",
        str(validated["fees_round_trip_pct"]),
        "--seed",
        str(validated["seed"]),
        "--run-stage4-simulate",
        str(validated["run_stage4_simulate"]),
    ]
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            shell=False,
        )
    acquire_lock(run_id=run_id, pid=process.pid, command=" ".join(cmd), runs_dir=runs_dir)
    return run_id, int(process.pid)


def start_whitelisted_script(
    script_relpath: str,
    args: list[str],
    runs_dir: Path = RUNS_DIR,
    run_id: str | None = None,
) -> tuple[str, int]:
    """Start a whitelisted script with explicit argument list."""

    script = validate_whitelisted_script(script_relpath)
    resolved_run_id = run_id or f"{utc_now_compact()}_{stable_hash({'script': script, 'args': args}, length=10)}"
    run_dir = Path(runs_dir) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "ui_stdout.log"
    stderr_path = run_dir / "ui_stderr.log"
    cmd = [sys.executable, script, *[str(item) for item in args]]
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            shell=False,
        )
    acquire_lock(run_id=resolved_run_id, pid=process.pid, command=" ".join(cmd), runs_dir=runs_dir)
    return resolved_run_id, int(process.pid)


def cancel_run(run_id: str, runs_dir: Path = RUNS_DIR) -> None:
    """Cancel run by writing cancel flag and terminating active process if present."""

    run_dir = Path(runs_dir) / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cancel.flag").write_text("cancelled by user\n", encoding="utf-8")

    active = get_active_run(runs_dir)
    if active is None or str(active.get("run_id")) != str(run_id):
        return
    pid = int(active.get("pid", 0))
    if pid <= 0:
        return
    _terminate_pid(pid)


def run_export_to_library(
    run_id: str,
    display_name: str | None = None,
    runs_dir: Path = RUNS_DIR,
    library_dir: Path = PROJECT_ROOT / "library",
) -> dict[str, Any]:
    """Run export_to_library script synchronously and return strategy card payload."""

    script = validate_whitelisted_script("scripts/export_to_library.py")
    args = [
        sys.executable,
        script,
        "--run-id",
        str(run_id),
        "--runs-dir",
        str(Path(runs_dir)),
        "--library-dir",
        str(Path(library_dir)),
    ]
    if display_name is not None and str(display_name).strip():
        args.extend(["--display-name", str(display_name).strip()])

    result = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "export_to_library failed")
    try:
        payload = json.loads(result.stdout)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse export_to_library output: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("export_to_library output is not a JSON object")
    return payload


def _terminate_pid(pid: int) -> None:
    try:
        import psutil  # type: ignore

        process = psutil.Process(int(pid))
        process.terminate()
        return
    except Exception:
        pass
    try:
        os.kill(int(pid), signal.SIGTERM)
    except Exception:
        pass
