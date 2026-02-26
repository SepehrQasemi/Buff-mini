"""Main Streamlit app entrypoint."""

from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="Buff-mini", page_icon="B", layout="wide")

st.title("Buff-mini")
st.caption("Auto Crypto Strategy Discovery Engine - MVP Phase 1")

st.sidebar.header("Navigation")
if hasattr(st.sidebar, "page_link"):
    st.sidebar.page_link("pages/1_dashboard.py", label="Dashboard")
    st.sidebar.page_link("pages/2_settings.py", label="Settings")
    st.sidebar.page_link("pages/3_run.py", label="Run")
    st.sidebar.page_link("pages/4_results.py", label="Results")

st.markdown(
    "Use the sidebar pages to inspect data, review settings, execute Stage-0, and view run results."
)

st.info("Discovery generator UI is intentionally disabled in MVP Phase 1.")
