"""Stage-3.1 Monte Carlo page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from buffmini.portfolio.monte_carlo import run_stage3_monte_carlo


root = Path(__file__).resolve().parents[4]
runs_dir = root / "runs"
data_dir = root / "data" / "raw"


def _latest_stage2_run_id() -> str:
    candidates = sorted([path for path in runs_dir.glob("*_stage2") if path.is_dir()], reverse=True)
    return candidates[0].name if candidates else ""


st.title("Stage-3.1 Monte Carlo")
st.write("Run deterministic Monte Carlo robustness simulation on an existing Stage-2 portfolio using cached local data only.")

stage2_run_id = st.text_input("Stage-2 run id", value=_latest_stage2_run_id())
methods = st.text_input("Methods", value="equal,vol,corr-min")
bootstrap = st.selectbox("Bootstrap type", options=["block", "iid"], index=0)
block_size = st.number_input("Block size (trades)", min_value=1, value=10, step=1)
n_paths = st.number_input("Number of paths", min_value=100, max_value=50000, value=20000, step=100)
initial_equity = st.number_input("Initial equity", min_value=1.0, value=10000.0, step=1000.0)
ruin_threshold = st.number_input("Ruin drawdown threshold", min_value=0.01, max_value=0.99, value=0.50, step=0.01, format="%.2f")
seed = st.number_input("Seed", min_value=0, value=42, step=1)
leverage = st.number_input("Leverage", min_value=0.1, value=1.0, step=0.1, format="%.2f")
save_paths = st.checkbox("Save per-path parquet files", value=False)

if st.button("Run Stage-3.1"):
    if not stage2_run_id.strip():
        st.error("Stage-2 run id is required.")
    else:
        resolved_methods = [item.strip() for item in methods.split(",") if item.strip()]
        run_dir = run_stage3_monte_carlo(
            stage2_run_id=stage2_run_id.strip(),
            methods=resolved_methods,
            bootstrap=bootstrap,
            block_size_trades=int(block_size),
            n_paths=int(n_paths),
            initial_equity=float(initial_equity),
            ruin_dd_threshold=float(ruin_threshold),
            seed=int(seed),
            leverage=float(leverage),
            save_paths=bool(save_paths),
            runs_dir=runs_dir,
            data_dir=data_dir,
        )
        summary = json.loads((run_dir / "mc_summary.json").read_text(encoding="utf-8"))

        results_rows = []
        tail_rows = []
        for method_key, payload in summary["methods"].items():
            mc = payload["summary"]
            results_rows.append(
                {
                    "method": method_key,
                    "trade_count_source": payload["trade_count_source"],
                    "return_p05": mc["return_pct"]["p05"],
                    "return_median": mc["return_pct"]["median"],
                    "return_p95": mc["return_pct"]["p95"],
                    "maxDD_p95": mc["max_drawdown"]["p95"],
                    "maxDD_p99": mc["max_drawdown"]["p99"],
                    "P_return_lt_0": mc["tail_probabilities"]["p_return_lt_0"],
                    "P_ruin": mc["tail_probabilities"]["p_ruin"],
                }
            )
            tail_rows.append(
                {
                    "method": method_key,
                    "P(maxDD>20%)": mc["tail_probabilities"]["p_maxdd_gt_20"],
                    "P(maxDD>30%)": mc["tail_probabilities"]["p_maxdd_gt_30"],
                    "P(maxDD>40%)": mc["tail_probabilities"]["p_maxdd_gt_40"],
                    "CVaR5": mc["return_pct"]["cvar5"],
                }
            )

        st.success(f"Stage-3.1 completed: {summary['run_id']}")
        st.write(f"Recommendation: `{summary['recommendation']}`")
        st.subheader("Results")
        st.dataframe(pd.DataFrame(results_rows))
        st.subheader("Tail probabilities")
        st.dataframe(pd.DataFrame(tail_rows))
        st.write(f"Report: `{run_dir / 'mc_report.md'}`")
