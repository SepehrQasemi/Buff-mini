"""Dashboard page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


root = project_root()
raw_dir = root / "data" / "raw"
runs_dir = root / "runs"

st.title("Dashboard")

files = sorted(raw_dir.glob("*.parquet")) if raw_dir.exists() else []
run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], reverse=True) if runs_dir.exists() else []

st.subheader("Data Status")
st.write(f"Parquet files: {len(files)}")
if files:
    st.dataframe({"file": [p.name for p in files]})

st.subheader("Previous Runs")
st.write(f"Run folders: {len(run_dirs)}")
if run_dirs:
    st.dataframe({"run_id": [p.name for p in run_dirs[:20]]})
