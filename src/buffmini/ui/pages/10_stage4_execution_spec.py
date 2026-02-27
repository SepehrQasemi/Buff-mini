"""Stage-4 execution controls, spec export, and paper simulation."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from buffmini.config import load_config
from buffmini.execution.simulator import resolve_stage4_method_and_leverage, run_stage4_simulation
from buffmini.spec.trading_spec import generate_trading_spec


root = Path(__file__).resolve().parents[4]
runs_dir = root / "runs"
data_dir = root / "data" / "raw"
default_config_path = root / "configs" / "default.yaml"


def _latest_run_id(suffix: str) -> str:
    candidates = sorted([path for path in runs_dir.glob(f"*_{suffix}") if path.is_dir()], reverse=True)
    return candidates[0].name if candidates else ""


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_stage1_candidates(stage1_run_id: str) -> dict[str, dict]:
    candidates_dir = runs_dir / stage1_run_id / "candidates"
    lookup: dict[str, dict] = {}
    for path in sorted(candidates_dir.glob("strategy_*.json")):
        payload = _load_json(path)
        lookup[str(payload["candidate_id"])] = payload
    return lookup


st.title("Stage-4 Execution Spec")
st.write("Configure execution policy and risk controls, export trading spec documents, and run offline paper simulation.")

cfg = load_config(default_config_path)
stage2_run_id = st.text_input("Stage-2 run id", value=_latest_run_id("stage2"))
stage3_run_id = st.text_input("Stage-3.3 run id (optional)", value=_latest_run_id("stage3_3_selector"))

st.subheader("Execution Policy")
cfg["execution"]["mode"] = st.selectbox("Mode", options=["net", "hedge", "isolated"], index=["net", "hedge", "isolated"].index(cfg["execution"]["mode"]))
cfg["execution"]["per_symbol_netting"] = st.checkbox("Per-symbol netting", value=bool(cfg["execution"]["per_symbol_netting"]))
cfg["execution"]["allow_opposite_signals"] = st.checkbox("Allow opposite signals", value=bool(cfg["execution"]["allow_opposite_signals"]))

st.subheader("Sizing & Caps")
cfg["risk"]["sizing"]["mode"] = st.selectbox("Sizing mode", options=["risk_budget", "fixed_fraction"], index=["risk_budget", "fixed_fraction"].index(cfg["risk"]["sizing"]["mode"]))
cfg["risk"]["sizing"]["risk_per_trade_pct"] = st.number_input("Risk per trade (%)", min_value=0.01, max_value=100.0, value=float(cfg["risk"]["sizing"]["risk_per_trade_pct"]), step=0.1)
cfg["risk"]["sizing"]["fixed_fraction_pct"] = st.number_input("Fixed fraction (%)", min_value=0.01, max_value=100.0, value=float(cfg["risk"]["sizing"]["fixed_fraction_pct"]), step=0.1)
cfg["risk"]["max_gross_exposure"] = st.number_input("Max gross exposure (x)", min_value=0.1, value=float(cfg["risk"]["max_gross_exposure"]), step=0.1)
cfg["risk"]["max_net_exposure_per_symbol"] = st.number_input("Max net exposure per symbol (x)", min_value=0.1, value=float(cfg["risk"]["max_net_exposure_per_symbol"]), step=0.1)
cfg["risk"]["max_open_positions"] = int(st.number_input("Max open positions", min_value=1, value=int(cfg["risk"]["max_open_positions"]), step=1))

st.subheader("Kill-Switch")
cfg["risk"]["killswitch"]["enabled"] = st.checkbox("Kill-switch enabled", value=bool(cfg["risk"]["killswitch"]["enabled"]))
cfg["risk"]["killswitch"]["max_daily_loss_pct"] = st.number_input("Max daily loss (%)", min_value=0.1, max_value=100.0, value=float(cfg["risk"]["killswitch"]["max_daily_loss_pct"]), step=0.1)
cfg["risk"]["killswitch"]["max_peak_to_valley_dd_pct"] = st.number_input("Max peak-to-valley DD (%)", min_value=0.1, max_value=100.0, value=float(cfg["risk"]["killswitch"]["max_peak_to_valley_dd_pct"]), step=0.1)
cfg["risk"]["killswitch"]["max_consecutive_losses"] = int(st.number_input("Max consecutive losses", min_value=1, value=int(cfg["risk"]["killswitch"]["max_consecutive_losses"]), step=1))
cfg["risk"]["killswitch"]["cool_down_bars"] = int(st.number_input("Cooldown bars", min_value=1, value=int(cfg["risk"]["killswitch"]["cool_down_bars"]), step=1))

days = int(st.number_input("Simulation days", min_value=1, value=90, step=1))
seed = int(st.number_input("Simulation seed", min_value=0, value=42, step=1))

stage3_summary = None
if stage3_run_id.strip():
    candidate_path = runs_dir / stage3_run_id.strip() / "selector_summary.json"
    if candidate_path.exists():
        stage3_summary = _load_json(candidate_path)

selected_method, selected_leverage, from_stage3, warnings = resolve_stage4_method_and_leverage(cfg=cfg, stage3_choice=stage3_summary)
st.info(
    f"Chosen method/leverage: `{selected_method}` @ `{selected_leverage}x` "
    f"(source: {'Stage-3.3' if from_stage3 else 'fallback defaults'})"
)
for warning in warnings:
    st.warning(warning)

if st.button("Generate Trading Spec"):
    if not stage2_run_id.strip():
        st.error("Stage-2 run id is required.")
    else:
        stage2_summary = _load_json(runs_dir / stage2_run_id.strip() / "portfolio_summary.json")
        method_payload = stage2_summary["portfolio_methods"][selected_method]
        weight_map = {str(candidate_id): float(weight) for candidate_id, weight in method_payload["weights"].items()}
        lookup = _load_stage1_candidates(str(stage2_summary["stage1_run_id"]))
        selected_candidates = []
        for candidate_id, weight in weight_map.items():
            if weight <= 0:
                continue
            payload = lookup.get(candidate_id)
            if payload is None:
                continue
            selected_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "strategy_family": str(payload["strategy_family"]),
                    "gating": str(payload["gating"]),
                    "exit_mode": str(payload["exit_mode"]),
                    "parameters": dict(payload["parameters"]),
                    "weight": float(weight),
                }
            )
        outputs = generate_trading_spec(
            cfg=cfg,
            stage2_metadata=stage2_summary,
            stage3_3_choice=stage3_summary,
            selected_candidates=selected_candidates,
            output_path=root / "docs" / "trading_spec.md",
        )
        st.success("Trading spec generated.")
        st.write(f"Trading spec: `{outputs['trading_spec_path']}`")
        st.write(f"Paper checklist: `{outputs['paper_checklist_path']}`")
        preview = outputs["trading_spec_path"].read_text(encoding="utf-8").splitlines()[:60]
        st.code("\n".join(preview), language="markdown")

if st.button("Run Paper Simulation"):
    if not stage2_run_id.strip():
        st.error("Stage-2 run id is required.")
    else:
        run_dir = run_stage4_simulation(
            stage2_run_id=stage2_run_id.strip(),
            cfg=cfg,
            stage3_choice=stage3_summary,
            days=days,
            bars=None,
            runs_dir=runs_dir,
            data_dir=data_dir,
            seed=seed,
        )
        metrics = _load_json(run_dir / "execution_metrics.json")
        st.success(f"Simulation completed: {run_dir.name}")
        st.json(metrics["metrics"])
        st.write(f"Artifacts: `{run_dir}`")

