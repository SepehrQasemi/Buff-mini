"""Stage-5 Run Monitor page."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from buffmini.constants import RUNS_DIR
from buffmini.ui.components.run_exec import cancel_run
from buffmini.ui.components.run_index import scan_runs
from buffmini.ui.components.run_lock import get_active_run


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return "(log file not found)"
    try:
        all_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as exc:
        return f"(failed to read log: {exc})"
    if not all_lines:
        return "(no logs yet)"
    return "\n".join(all_lines[-lines:])


def _cpu_mem() -> tuple[float | None, float | None]:
    try:
        import psutil  # type: ignore

        return float(psutil.cpu_percent(interval=0.0)), float(psutil.virtual_memory().percent)
    except Exception:
        return None, None


def _timeline(stage: str, stage_idx: int, stage_total: int) -> list[tuple[str, str]]:
    names = ["data_validate", "stage1", "stage2", "stage3_3", "stage4_spec", "stage4_sim"]
    ordered = names[: max(stage_total, 0)]
    rows: list[tuple[str, str]] = []
    for idx, name in enumerate(ordered, start=1):
        if idx < stage_idx:
            status = "done"
        elif idx == stage_idx:
            status = "running" if stage != "done" else "done"
        else:
            status = "pending"
        rows.append((name, status))
    return rows


st.title("Stage-5 Run Monitor")
active = get_active_run()
selected_run_id = str(st.session_state.get("last_pipeline_run_id", ""))

if active:
    run_id = str(active.get("run_id"))
    st.success(f"Connected to active run: `{run_id}`")
else:
    run_id = selected_run_id
    if run_id:
        st.info(f"No active lock found. Showing last selected run: `{run_id}`")

if not run_id:
    runs = scan_runs(RUNS_DIR)
    if not runs:
        st.info("No runs found yet.")
        st.stop()
    run_id = str(runs[0]["run_id"])
    st.info(f"Showing most recent run: `{run_id}`")

run_dir = Path(RUNS_DIR) / run_id
progress = _read_json(run_dir / "progress.json")
pipeline_summary = _read_json(run_dir / "pipeline_summary.json")

stage = str(progress.get("stage", pipeline_summary.get("status", "unknown")))
stage_idx = int(progress.get("stage_idx", 0) or 0)
stage_total = int(progress.get("stage_total", 0) or 0)
status = str(progress.get("status", pipeline_summary.get("status", "unknown")))

left, right = st.columns(2)
with left:
    st.metric("Status", status)
    st.metric("Stage", stage)
with right:
    st.metric("Elapsed (s)", round(float(progress.get("elapsed_seconds", 0.0) or 0.0), 1))
    st.metric("ETA (s)", round(float(progress.get("eta_seconds", 0.0) or 0.0), 1))

if stage_total > 0:
    overall_pct = ((max(stage_idx - 1, 0) + float(progress.get("stage_progress_pct", 0.0)) / 100.0) / stage_total) * 100.0
    st.progress(int(max(0.0, min(100.0, overall_pct))), text=f"Overall {overall_pct:.1f}%")

st.subheader("Stage Timeline")
for name, item_status in _timeline(stage=stage, stage_idx=stage_idx, stage_total=stage_total):
    st.write(f"- `{name}`: **{item_status}**")

st.subheader("Live Counters")
st.json(progress.get("counters", {}))

cpu, mem = _cpu_mem()
if cpu is None:
    st.caption("CPU/RAM unavailable (psutil not installed).")
else:
    st.caption(f"CPU: {cpu:.1f}% | RAM: {mem:.1f}%")

st.subheader("Log Tail")
st.code(_tail(run_dir / "ui_stdout.log", lines=140), language="text")

col1, col2 = st.columns(2)
with col1:
    if st.button("Cancel Run"):
        try:
            cancel_run(run_id=run_id)
            st.warning("Cancellation requested.")
        except Exception as exc:
            st.error(f"Failed to cancel run: {exc}")
with col2:
    if st.button("Send To Background") and hasattr(st, "switch_page"):
        st.switch_page("pages/22_results_studio.py")

if status in {"done", "success"}:
    st.success("Run completed.")
    if st.button("Open Results") and hasattr(st, "switch_page"):
        st.switch_page("pages/22_results_studio.py")
elif status == "running":
    time.sleep(2)
    st.rerun()
