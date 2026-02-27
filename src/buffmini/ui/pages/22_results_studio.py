"""Stage-5 Results Studio page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from buffmini.constants import PROJECT_ROOT, RUNS_DIR
from buffmini.ui.components.artifacts import (
    load_generic_curves,
    load_pipeline_summary,
    load_stage3_2_artifacts,
    load_stage3_3_artifacts,
    load_stage4_artifacts,
)
from buffmini.ui.components.charts import plot_leverage_frontier, plot_mc_quantiles, plot_selector_log_growth
from buffmini.ui.components.library import export_run_to_library
from buffmini.ui.components.markdown_viewer import render_markdown_file
from buffmini.ui.components.run_exec import start_whitelisted_script
from buffmini.ui.components.run_index import latest_completed_pipeline, scan_runs
from buffmini.ui.components.trade_map import load_bundle_summary, plot_trade_map


st.title("Stage-5 Results Studio")
st.write("Artifact-driven run explorer. No heavy recomputation is performed for display.")

run_records = scan_runs(RUNS_DIR)
if not run_records:
    st.info("No runs found.")
    st.stop()

default_run = latest_completed_pipeline(RUNS_DIR)
default_run_id = default_run["run_id"] if default_run else run_records[0]["run_id"]
selected_run_id = st.selectbox("Run ID", options=[item["run_id"] for item in run_records], index=[item["run_id"] for item in run_records].index(default_run_id))
run_dir = Path(RUNS_DIR) / selected_run_id

pipeline_payload, pipeline_warnings = load_pipeline_summary(run_dir)
pipeline_summary = pipeline_payload.get("pipeline_summary", {})
progress = pipeline_payload.get("progress", {})

stage3_2_run_id = pipeline_summary.get("stage3_2_run_id")
stage3_3_run_id = pipeline_summary.get("stage3_3_run_id")
stage4_sim_run_id = pipeline_summary.get("stage4_sim_run_id")
stage4_run_id = pipeline_summary.get("stage4_run_id")

stage3_2_dir = (Path(RUNS_DIR) / str(stage3_2_run_id)) if stage3_2_run_id else run_dir
stage3_3_dir = (Path(RUNS_DIR) / str(stage3_3_run_id)) if stage3_3_run_id else run_dir
stage4_dir = (Path(RUNS_DIR) / str(stage4_sim_run_id or stage4_run_id)) if (stage4_sim_run_id or stage4_run_id) else run_dir

stage3_2_payload, stage3_2_warnings = load_stage3_2_artifacts(stage3_2_dir)
stage3_3_payload, stage3_3_warnings = load_stage3_3_artifacts(stage3_3_dir)
stage4_payload, stage4_warnings = load_stage4_artifacts(stage4_dir, project_root=PROJECT_ROOT)
generic_curves, generic_warnings = load_generic_curves(run_dir)

all_warnings = pipeline_warnings + stage3_2_warnings + stage3_3_warnings + stage4_warnings + generic_warnings
if all_warnings:
    with st.expander("Artifact warnings"):
        for warning in all_warnings:
            st.write(f"- {warning}")

summary_tab, charts_tab, trade_tab, exposure_tab, reports_tab = st.tabs(
    ["Summary", "Charts", "Trade Map", "Exposure & Risk", "Reports"]
)

with summary_tab:
    st.subheader("Pipeline Summary")
    if pipeline_summary:
        st.json(
            {
                "status": pipeline_summary.get("status"),
                "stage1_run_id": pipeline_summary.get("stage1_run_id"),
                "stage2_run_id": pipeline_summary.get("stage2_run_id"),
                "stage3_3_run_id": pipeline_summary.get("stage3_3_run_id"),
                "chosen_method": pipeline_summary.get("chosen_method"),
                "chosen_leverage": pipeline_summary.get("chosen_leverage"),
                "elapsed_seconds": pipeline_summary.get("elapsed_seconds"),
            }
        )
    else:
        st.info("pipeline_summary.json not available for this run.")
    if progress:
        st.caption(f"Last stage: {progress.get('stage')} | status: {progress.get('status')}")

    st.subheader("Save To Library")
    display_name = st.text_input("Display name (optional)", value="")
    if st.button("Save to Library"):
        try:
            card = export_run_to_library(run_id=selected_run_id, display_name=display_name or None)
        except Exception as exc:
            st.error(f"Export failed: {exc}")
        else:
            st.success(f"Saved strategy `{card['strategy_id']}` to library.")
            st.json(card)

with charts_tab:
    st.subheader("Leverage Frontier")
    frontier_df = stage3_2_payload.get("table", pd.DataFrame())
    fig_frontier = plot_leverage_frontier(frontier_df)
    if fig_frontier is None:
        st.info("Frontier chart not available for this run.")
    else:
        st.pyplot(fig_frontier)

    st.subheader("Selector Log-Growth Curve")
    selector_df = stage3_3_payload.get("table", pd.DataFrame())
    fig_selector = plot_selector_log_growth(selector_df)
    if fig_selector is None:
        st.info("Selector chart not available for this run.")
    else:
        st.pyplot(fig_selector)

    st.subheader("Monte Carlo Quantiles")
    quantiles = pd.DataFrame()
    if stage3_2_payload.get("summary"):
        # Stage-3.2 does not persist MC quantiles table; this stays optional.
        quantiles = pd.DataFrame()
    fig_mc = plot_mc_quantiles(quantiles)
    if fig_mc is None:
        st.info("MC quantiles chart not available.")
    else:
        st.pyplot(fig_mc)

with trade_tab:
    st.subheader("Trade Map")
    bundle_summary, bundle_warnings = load_bundle_summary(run_dir)
    for warning in bundle_warnings:
        st.caption(warning)

    bundle_trades = run_dir / "ui_bundle" / "trades.csv"
    if not bundle_trades.exists():
        st.info("ui_bundle/trades.csv is missing for this run.")
        stage2_for_run = pipeline_summary.get("stage2_run_id")
        if stage2_for_run:
            if st.button("Run Stage-4 simulation for this run"):
                args = ["--stage2-run-id", str(stage2_for_run)]
                pipeline_cfg = run_dir / "pipeline_config.yaml"
                if pipeline_cfg.exists():
                    args.extend(["--config", str(pipeline_cfg)])
                stage3_for_run = pipeline_summary.get("stage3_3_run_id")
                if stage3_for_run:
                    args.extend(["--stage3-3-run-id", str(stage3_for_run)])
                try:
                    launched_run_id, pid = start_whitelisted_script(
                        script_relpath="scripts/run_stage4_simulate.py",
                        args=args,
                    )
                except Exception as exc:
                    st.error(f"Failed to start Stage-4 simulation: {exc}")
                else:
                    st.success(f"Started Stage-4 simulation helper run `{launched_run_id}` (pid={pid}).")
        else:
            st.caption("No stage2_run_id found in pipeline summary.")

    symbol = st.selectbox("Symbol", options=["BTC/USDT", "ETH/USDT"], index=0)
    direction_filter = st.selectbox("Direction", options=["both", "long", "short"], index=0)
    fig_trade, plot_warnings, marker_frame = plot_trade_map(
        run_dir=run_dir,
        symbol=symbol,
        direction_filter=direction_filter,
    )
    for warning in plot_warnings:
        st.caption(warning)
    if fig_trade is None:
        st.info("Trade map unavailable. If no trades artifact exists, run Stage-4 simulation for this run.")
    else:
        st.pyplot(fig_trade)
        st.caption(f"Markers: {len(marker_frame)}")

with exposure_tab:
    st.subheader("Exposure Time Series")
    exposure_df = stage4_payload.get("exposure_timeseries", pd.DataFrame())
    if exposure_df.empty:
        st.info("No exposure_timeseries.csv found.")
    else:
        st.dataframe(exposure_df.tail(500))

    st.subheader("Kill-switch Events")
    ks_df = stage4_payload.get("killswitch_events", pd.DataFrame())
    if ks_df.empty:
        st.info("No kill-switch events file found.")
    else:
        st.dataframe(ks_df)

with reports_tab:
    st.subheader("Markdown Reports")
    reports = pipeline_summary.get("reports", {}) if pipeline_summary else {}

    selector_report = Path(reports.get("stage3_3_report", stage3_3_dir / "selector_report.md"))
    render_markdown_file(selector_report, title="Stage-3.3 Selector Report", fallback="Selector report unavailable")

    stage2_report = Path(reports.get("stage2_report", Path(RUNS_DIR) / str(pipeline_summary.get("stage2_run_id", "")) / "portfolio_report.md"))
    render_markdown_file(stage2_report, title="Stage-2 Portfolio Report", fallback="Stage-2 report unavailable")

    trading_spec_report = Path(
        reports.get("trading_spec", stage4_dir / "spec" / "trading_spec.md")
    )
    checklist_report = Path(
        reports.get("paper_checklist", stage4_dir / "spec" / "paper_trading_checklist.md")
    )
    render_markdown_file(trading_spec_report, title="Trading Spec", fallback="trading_spec.md not found")
    render_markdown_file(checklist_report, title="Paper Trading Checklist", fallback="paper_trading_checklist.md not found")

    if not generic_curves.get("equity_curve", pd.DataFrame()).empty:
        st.subheader("Equity Curve (generic artifact)")
        st.dataframe(generic_curves["equity_curve"].tail(200))
