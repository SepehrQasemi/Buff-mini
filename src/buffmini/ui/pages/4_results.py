"""Results page (placeholder)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


root = Path(__file__).resolve().parents[4]
runs_dir = root / "runs"

st.title("Results")

run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], reverse=True) if runs_dir.exists() else []
if not run_dirs:
    st.info("No run artifacts found yet.")
else:
    latest = run_dirs[0]
    summary_path = latest / "summary.csv"
    st.write(f"Latest run: {latest.name}")
    if summary_path.exists():
        st.dataframe(pd.read_csv(summary_path))
    else:
        st.info("Summary file not found for latest run (placeholder page).")
