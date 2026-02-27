"""Stage-5 run compare page."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from buffmini.constants import RUNS_DIR
from buffmini.ui.components.run_compare import (
    build_comparison_table,
    load_ui_bundle_charts_index,
    load_ui_bundle_curve,
    load_ui_bundle_summary,
    resolve_run_metrics,
)
from buffmini.ui.components.run_index import scan_runs


st.title("Stage-5 Run Compare")
st.write("Compare two runs side-by-side using only standardized ui_bundle artifacts.")

records = scan_runs(RUNS_DIR)
if len(records) < 2:
    st.info("Need at least two runs to compare.")
    st.stop()

run_ids = [item["run_id"] for item in records]
default_a = 0
default_b = 1 if len(run_ids) > 1 else 0
run_a_id = st.selectbox("Run A", options=run_ids, index=default_a)
run_b_id = st.selectbox("Run B", options=run_ids, index=default_b)

run_a_dir = Path(RUNS_DIR) / run_a_id
run_b_dir = Path(RUNS_DIR) / run_b_id

summary_a, warn_a = load_ui_bundle_summary(run_a_dir)
summary_b, warn_b = load_ui_bundle_summary(run_b_dir)

for warning in warn_a + warn_b:
    st.caption(warning)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Run A")
    st.json(
        {
            "run_id": run_a_id,
            "status": summary_a.get("status"),
            "method": summary_a.get("chosen_method"),
            "leverage": summary_a.get("chosen_leverage"),
            "execution_mode": summary_a.get("execution_mode"),
        }
    )
with col2:
    st.subheader("Run B")
    st.json(
        {
            "run_id": run_b_id,
            "status": summary_b.get("status"),
            "method": summary_b.get("chosen_method"),
            "leverage": summary_b.get("chosen_leverage"),
            "execution_mode": summary_b.get("execution_mode"),
        }
    )

st.subheader("Metrics Comparison")
compare_table, compare_warnings = build_comparison_table(run_a_dir, run_b_dir)
for warning in compare_warnings:
    st.caption(warning)
st.dataframe(compare_table)

st.subheader("Equity Overlay")
curve_a, curve_warn_a = load_ui_bundle_curve(run_a_dir, "equity_curve.csv")
curve_b, curve_warn_b = load_ui_bundle_curve(run_b_dir, "equity_curve.csv")
for warning in curve_warn_a + curve_warn_b:
    st.caption(warning)
if curve_a.empty or curve_b.empty or "timestamp" not in curve_a.columns or "timestamp" not in curve_b.columns:
    st.info("Equity overlay not available for both runs.")
else:
    fig, ax = plt.subplots(figsize=(10, 4))
    series_a = curve_a.copy()
    series_b = curve_b.copy()
    series_a["timestamp"] = pd.to_datetime(series_a["timestamp"], utc=True, errors="coerce")
    series_b["timestamp"] = pd.to_datetime(series_b["timestamp"], utc=True, errors="coerce")
    series_a = series_a.dropna(subset=["timestamp"]).sort_values("timestamp")
    series_b = series_b.dropna(subset=["timestamp"]).sort_values("timestamp")
    ax.plot(series_a["timestamp"], pd.to_numeric(series_a["equity"], errors="coerce"), label=run_a_id)
    ax.plot(series_b["timestamp"], pd.to_numeric(series_b["equity"], errors="coerce"), label=run_b_id)
    ax.set_title("Equity Curve Overlay")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("equity")
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

st.subheader("Leverage Curves")
charts_a, chart_warn_a = load_ui_bundle_charts_index(run_a_dir)
charts_b, chart_warn_b = load_ui_bundle_charts_index(run_b_dir)
for warning in chart_warn_a + chart_warn_b:
    st.caption(warning)

selector_a = charts_a.get("selector_table") if isinstance(charts_a, dict) else None
selector_b = charts_b.get("selector_table") if isinstance(charts_b, dict) else None

if selector_a and selector_b:
    try:
        table_a = pd.read_csv(selector_a)
        table_b = pd.read_csv(selector_b)
    except Exception as exc:
        st.info(f"Leverage curve table parse failed: {exc}")
    else:
        if {"leverage", "expected_log_growth"}.issubset(table_a.columns) and {"leverage", "expected_log_growth"}.issubset(table_b.columns):
            metrics_a, _ = resolve_run_metrics(run_a_dir)
            metrics_b, _ = resolve_run_metrics(run_b_dir)
            method_a = str(metrics_a.get("chosen_method", ""))
            method_b = str(metrics_b.get("chosen_method", ""))

            if method_a and "method" in table_a.columns:
                table_a = table_a[table_a["method"].astype(str) == method_a]
            if method_b and "method" in table_b.columns:
                table_b = table_b[table_b["method"].astype(str) == method_b]

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(table_a["leverage"], table_a["expected_log_growth"], marker="o", label=f"{run_a_id}")
            ax2.plot(table_b["leverage"], table_b["expected_log_growth"], marker="o", label=f"{run_b_id}")
            ax2.set_title("Expected Log Growth vs Leverage")
            ax2.set_xlabel("leverage")
            ax2.set_ylabel("expected_log_growth")
            ax2.legend(loc="best")
            fig2.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("Selector tables missing required columns for leverage overlay.")
else:
    st.info("Selector leverage curves unavailable for one or both runs.")
