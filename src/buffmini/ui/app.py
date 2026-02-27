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
    st.sidebar.page_link("pages/5_auto_optimize.py", label="Auto Optimize")
    st.sidebar.page_link("pages/6_stage2_5_walkforward.py", label="Stage-2.7 Walk-Forward")
    st.sidebar.page_link("pages/6_stage2_probabilistic.py", label="Stage-2.8 Probabilistic")
    st.sidebar.page_link("pages/7_stage3_monte_carlo.py", label="Stage-3.1 Monte Carlo")
    st.sidebar.page_link("pages/9_stage3_leverage_selector.py", label="Stage-3.3 Leverage Selector")
    st.sidebar.page_link("pages/10_stage4_execution_spec.py", label="Stage-4 Execution")
    st.sidebar.page_link("pages/20_strategy_lab.py", label="Stage-5 Strategy Lab")
    st.sidebar.page_link("pages/21_run_monitor.py", label="Stage-5 Run Monitor")
    st.sidebar.page_link("pages/22_results_studio.py", label="Stage-5 Results Studio")
    st.sidebar.page_link("pages/23_strategy_library.py", label="Stage-5 Strategy Library")
    st.sidebar.page_link("pages/24_run_compare.py", label="Stage-5 Run Compare")

st.markdown(
    "Use the sidebar pages to inspect data, review settings, execute Stage-0, run Stage-1 optimization, validate Stage-2.7 walk-forward robustness, run Stage-2.8 probabilistic evidence checks, run Stage-3.1 Monte Carlo robustness analysis, run Stage-3.3 leverage selection, run Stage-4 execution/spec workflows, and use Stage-5 product UI (Strategy Lab, Run Monitor, Results Studio, Strategy Library, Run Compare)."
)

st.info("Stage-1 Auto Optimize is available in the dedicated page. Live trading is out of scope.")
