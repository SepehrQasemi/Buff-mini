"""Stage-2.8 probabilistic walk-forward page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from buffmini.portfolio.probabilistic import run_stage2_probabilistic


root = Path(__file__).resolve().parents[4]
runs_dir = root / "runs"
data_dir = root / "data" / "raw"


def _latest_stage2_run_id() -> str:
    candidates = sorted([path for path in runs_dir.glob("*_stage2") if path.is_dir()], reverse=True)
    return candidates[0].name if candidates else ""


st.title("Stage-2.8 Probabilistic Validator")
st.write("Run bootstrap-based rolling evidence analysis on an existing Stage-2 portfolio using cached local data only.")

stage2_run_id = st.text_input("Stage-2 run id", value=_latest_stage2_run_id())
window_days = st.number_input("Window days", min_value=1, value=30, step=1)
stride_days = st.number_input("Stride days", min_value=1, value=7, step=1)
reserve_tail_days = st.number_input("Reserve tail days", min_value=1, value=180, step=1)
num_windows_text = st.text_input("Num windows (optional)", value="")
min_trades = st.number_input("Min trades", min_value=0, value=20, step=1)
min_exposure = st.number_input("Min exposure", min_value=0.0, value=0.01, step=0.01, format="%.4f")
n_boot = st.number_input("Bootstrap samples", min_value=100, value=5000, step=100)
seed = st.number_input("Seed", min_value=0, value=42, step=1)

if st.button("Run Stage-2.8"):
    if not stage2_run_id:
        st.error("Stage-2 run id is required.")
    else:
        parsed_num_windows = int(num_windows_text) if num_windows_text.strip() else None
        command = (
            "python scripts/run_stage2_probabilistic.py "
            f"--stage2-run-id {stage2_run_id} --window-days {int(window_days)} "
            f"--stride-days {int(stride_days)} --reserve-tail-days {int(reserve_tail_days)} "
            f"--min_trades {int(min_trades)} --min_exposure {float(min_exposure)} "
            f"--n_boot {int(n_boot)} --seed {int(seed)}"
        )
        if parsed_num_windows is not None:
            command += f" --num-windows {parsed_num_windows}"
        run_dir = run_stage2_probabilistic(
            stage2_run_id=stage2_run_id,
            window_days=int(window_days),
            stride_days=int(stride_days),
            num_windows=parsed_num_windows,
            reserve_tail_days=int(reserve_tail_days),
            min_trades=int(min_trades),
            min_exposure=float(min_exposure),
            n_boot=int(n_boot),
            seed=int(seed),
            runs_dir=runs_dir,
            data_dir=data_dir,
            cli_command=command,
        )
        summary = json.loads((run_dir / "probabilistic_summary.json").read_text(encoding="utf-8"))
        rows = []
        for method_key in ["equal", "vol", "corr-min"]:
            payload = summary["method_summaries"].get(method_key)
            if payload is None:
                continue
            aggregate = payload["aggregate"]
            rows.append(
                {
                    "method": method_key,
                    "usable_windows": f"{aggregate['usable_windows']}/{aggregate['total_windows']}",
                    "p_edge_gt0_median": aggregate["p_edge_gt0_median"],
                    "p_pf_gt1_median": aggregate["p_pf_gt1_median"],
                    "robustness_score": aggregate["robustness_score"],
                    "classification": aggregate["classification"],
                }
            )

        st.success(f"Stage-2.8 completed: {summary['run_id']}")
        st.write(f"Recommendation: `{summary['overall_recommendation']}`")
        st.dataframe(pd.DataFrame(rows))
        st.write(f"Report: `{run_dir / 'probabilistic_report.md'}`")

        try:
            import matplotlib.pyplot as plt

            window_metrics = pd.read_csv(run_dir / "window_metrics_full.csv")
            usable = window_metrics.loc[window_metrics["usable"] == True].copy()
            if not usable.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                usable["p_edge_gt0"].astype(float).hist(ax=ax, bins=10)
                ax.set_title("Usable-window p_edge_gt0 distribution")
                ax.set_xlabel("p_edge_gt0")
                ax.set_ylabel("count")
                st.pyplot(fig)
            else:
                st.info("No usable windows available for p_edge_gt0 histogram.")
        except Exception:
            st.info("Matplotlib histogram unavailable in this environment; report and CSV artifacts were still generated.")
