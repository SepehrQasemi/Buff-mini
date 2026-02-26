"""Run page."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st


root = Path(__file__).resolve().parents[4]

st.title("Run Stage-0")
st.write("Execute Stage-0 baseline backtests using current config and local data.")

if st.button("Run Stage-0"):
    command = [sys.executable, "scripts/run_stage0.py"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    st.code(result.stdout if result.stdout else "(no output)")
    if result.returncode != 0:
        st.error(result.stderr if result.stderr else "Stage-0 failed.")
    else:
        st.success("Stage-0 completed.")
