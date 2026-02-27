"""Stage-3.3 leverage selector page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from buffmini.config import load_config
from buffmini.portfolio.leverage_selector import run_stage3_leverage_selector


root = Path(__file__).resolve().parents[4]
runs_dir = root / "runs"
data_dir = root / "data" / "raw"
config = load_config(root / "configs" / "default.yaml")
defaults = config["portfolio"]["leverage_selector"]


def _latest_stage2_run_id() -> str:
    candidates = sorted([path for path in runs_dir.glob("*_stage2") if path.is_dir()], reverse=True)
    return candidates[0].name if candidates else ""


st.title("Stage-3.3 Automatic Leverage Selector")
st.write("Choose leverage automatically under hard risk constraints, maximizing expected log-growth over feasible leverage levels.")

stage2_run_id = st.text_input("Stage-2 run id", value=_latest_stage2_run_id())
methods = st.text_input("Methods", value=",".join(defaults["methods"]))
leverage_levels = st.text_input("Leverage levels", value=",".join(str(item) for item in defaults["leverage_levels"]))
n_paths = st.number_input("Number of paths", min_value=1000, max_value=50000, value=int(defaults["n_paths"]), step=1000)
bootstrap = st.selectbox("Bootstrap", options=["block", "iid"], index=0 if defaults["bootstrap"] == "block" else 1)
block_size = st.number_input("Block size (trades)", min_value=1, value=int(defaults["block_size_trades"]), step=1)
seed = st.number_input("Seed", min_value=0, value=int(defaults["seed"]), step=1)
initial_equity = st.number_input("Initial equity", min_value=1.0, value=float(defaults["initial_equity"]), step=1000.0)
ruin_threshold = st.number_input("Ruin DD threshold", min_value=0.01, max_value=0.99, value=float(defaults["ruin_dd_threshold"]), step=0.01, format="%.2f")

st.subheader("Hard Constraints")
max_p_ruin = st.number_input("max_p_ruin", min_value=0.0, max_value=1.0, value=float(defaults["constraints"]["max_p_ruin"]), step=0.001, format="%.3f")
max_dd_p95 = st.number_input("max_dd_p95", min_value=0.0, max_value=1.0, value=float(defaults["constraints"]["max_dd_p95"]), step=0.01, format="%.2f")
min_return_p05 = st.number_input("min_return_p05", value=float(defaults["constraints"]["min_return_p05"]), step=0.01, format="%.2f")

if st.button("Run Stage-3.3"):
    if not stage2_run_id.strip():
        st.error("Stage-2 run id is required.")
    else:
        run_dir = run_stage3_leverage_selector(
            stage2_run_id=stage2_run_id.strip(),
            selector_cfg=defaults,
            methods=[item.strip() for item in methods.split(",") if item.strip()],
            leverage_levels=[float(item.strip()) for item in leverage_levels.split(",") if item.strip()],
            n_paths=int(n_paths),
            bootstrap=str(bootstrap),
            block_size_trades=int(block_size),
            seed=int(seed),
            initial_equity=float(initial_equity),
            ruin_dd_threshold=float(ruin_threshold),
            max_p_ruin=float(max_p_ruin),
            max_dd_p95=float(max_dd_p95),
            min_return_p05=float(min_return_p05),
            runs_dir=runs_dir,
            data_dir=data_dir,
        )
        summary = json.loads((run_dir / "selector_summary.json").read_text(encoding="utf-8"))
        table = pd.read_csv(run_dir / "selector_table.csv")

        method_rows = []
        for method, payload in summary["method_choices"].items():
            chosen = payload.get("chosen_row")
            method_rows.append(
                {
                    "method": method,
                    "chosen_leverage": payload.get("chosen_leverage"),
                    "expected_log_growth": None if chosen is None else chosen.get("expected_log_growth"),
                    "binding_constraints": payload.get("binding_constraints", payload.get("first_failure_constraints", [])),
                    "status": payload.get("status"),
                }
            )

        st.success(f"Stage-3.3 completed: {summary['run_id']}")
        st.subheader("Chosen Leverage Per Method")
        st.dataframe(pd.DataFrame(method_rows))
        st.subheader("Overall Choice")
        st.json(summary["overall_choice"])
        st.subheader("Expected Log-Growth by Leverage")
        st.dataframe(table[["method", "leverage", "expected_log_growth", "return_p05", "maxdd_p95", "p_ruin", "pass_all_constraints"]])
        st.write(f"Report: `{run_dir / 'selector_report.md'}`")

