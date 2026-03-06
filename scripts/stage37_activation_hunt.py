"""Stage-37.1 activation hunt: reject-chain diagnostics and threshold calibration."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage37.activation import (
    ActivationHuntConfig,
    calibrate_thresholds,
    compute_reject_chain_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-37 activation-hunt diagnostics")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="both", choices=["research", "live", "both"])
    parser.add_argument("--budget-small", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stage28-run-id", type=str, default="")
    return parser.parse_args()


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _run_stage28(*, config: Path, seed: int, mode: str, budget_small: bool, runs_dir: Path, docs_dir: Path) -> tuple[str, str, str]:
    cmd: list[str] = [
        sys.executable,
        "scripts/run_stage28.py",
        "--config",
        str(config),
        "--seed",
        str(int(seed)),
        "--mode",
        str(mode),
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
    ]
    if bool(budget_small):
        cmd.append("--budget-small")
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    stdout = str(proc.stdout or "")
    stderr = str(proc.stderr or "")
    if int(proc.returncode) != 0:
        tail = "\n".join((stdout + "\n" + stderr).splitlines()[-40:])
        raise RuntimeError(f"run_stage28 failed with exit={proc.returncode}\n{tail}")
    match = re.search(r"^run_id:\s*(\S+)\s*$", stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("Unable to parse run_id from run_stage28 output")
    return str(match.group(1)), stdout, stderr


def _stage37_cfg(config: dict[str, Any]) -> dict[str, Any]:
    stage37 = dict(((config.get("evaluation", {}) or {}).get("stage37", {})))
    return dict(stage37.get("activation_hunt", {}))


def _render_markdown(payload: dict[str, Any]) -> str:
    strict = dict(payload.get("strict", {}))
    hunt = dict(payload.get("hunt", {}))
    strict_overall = dict(strict.get("overall", {}))
    hunt_overall = dict(hunt.get("overall", {}))
    killer = dict(payload.get("main_killer", {}))
    lines = [
        "# Stage-37 Activation Hunt Report",
        "",
        "## Run Context",
        f"- stage28_run_id: `{payload.get('stage28_run_id', '')}`",
        f"- seed: `{int(payload.get('seed', 0))}`",
        f"- mode: `{payload.get('mode', '')}`",
        f"- budget_small: `{bool(payload.get('budget_small', False))}`",
        "",
        "## Strict vs Hunt",
        "| metric | strict | hunt |",
        "| --- | ---: | ---: |",
        f"| raw_signal_count | {int(strict_overall.get('raw_signal_count', 0))} | {int(hunt_overall.get('raw_signal_count', 0))} |",
        f"| post_threshold_count | {int(strict_overall.get('post_threshold_count', 0))} | {int(hunt_overall.get('post_threshold_count', 0))} |",
        f"| post_cost_gate_count | {int(strict_overall.get('post_cost_gate_count', 0))} | {int(hunt_overall.get('post_cost_gate_count', 0))} |",
        f"| post_feasibility_count | {int(strict_overall.get('post_feasibility_count', 0))} | {int(hunt_overall.get('post_feasibility_count', 0))} |",
        f"| final_trade_count | {float(strict_overall.get('final_trade_count', 0.0)):.3f} | {float(hunt_overall.get('final_trade_count', 0.0)):.3f} |",
        f"| activation_rate | {float(strict_overall.get('activation_rate', 0.0)):.6f} | {float(hunt_overall.get('activation_rate', 0.0)):.6f} |",
        "",
        "## Gate Killer",
        f"- dominant_gate: `{killer.get('gate', 'unknown')}`",
        f"- dropped_count: `{int(killer.get('dropped_count', 0))}`",
        "",
        "## Threshold Sensitivity",
        "| threshold | selected_rows | post_feasibility_count | avg_context_quality |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("threshold_curve", []):
        lines.append(
            "| {t:.4f} | {s} | {f} | {q:.6f} |".format(
                t=float(row.get("threshold", 0.0)),
                s=int(row.get("selected_rows", 0)),
                f=int(row.get("post_feasibility_count", 0)),
                q=float(row.get("avg_context_quality", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## Top Reject Reasons (Strict)",
        ]
    )
    reasons = dict(strict_overall.get("top_reject_reasons", {}))
    if reasons:
        for key, value in sorted(reasons.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            lines.append(f"- {key}: {int(value)}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Family Gate Summary (Hunt)",
            "| family | raw | post_threshold | post_cost | post_feasibility | activation_rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    per_family = dict(hunt.get("per_family", {}))
    for family in sorted(per_family.keys()):
        row = dict(per_family.get(family, {}))
        lines.append(
            "| {family} | {raw} | {thr} | {cost} | {feas} | {rate:.6f} |".format(
                family=family,
                raw=int(row.get("raw_signal_count", 0)),
                thr=int(row.get("post_threshold_count", 0)),
                cost=int(row.get("post_cost_gate_count", 0)),
                feas=int(row.get("post_feasibility_count", 0)),
                rate=float(row.get("activation_rate", 0.0)),
            )
        )
    return "\n".join(lines).strip() + "\n"


def _main_gate_killer(overall: dict[str, Any]) -> dict[str, Any]:
    raw = int(overall.get("raw_signal_count", 0))
    post_threshold = int(overall.get("post_threshold_count", 0))
    post_cost = int(overall.get("post_cost_gate_count", 0))
    post_feasible = int(overall.get("post_feasibility_count", 0))
    final_count = int(round(float(overall.get("final_trade_count", 0.0))))
    drops = [
        ("threshold", raw - post_threshold),
        ("cost_gate", post_threshold - post_cost),
        ("feasibility", post_cost - post_feasible),
        ("execution_to_trade", post_feasible - final_count),
    ]
    drops = [(name, max(0, value)) for name, value in drops]
    gate, count = sorted(drops, key=lambda item: (int(item[1]), str(item[0])), reverse=True)[0]
    return {"gate": str(gate), "dropped_count": int(count)}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    hunt_cfg = _stage37_cfg(cfg)
    threshold_grid = tuple(float(v) for v in hunt_cfg.get("threshold_grid", ActivationHuntConfig().threshold_grid))
    activation_cfg = ActivationHuntConfig(
        threshold_grid=threshold_grid,
        quality_floor=float(hunt_cfg.get("quality_floor", 0.0)),
        min_quality_floor=float(hunt_cfg.get("min_quality_floor", -0.02)),
        min_selected_rows=int(hunt_cfg.get("min_selected_rows", 1)),
    )

    stage28_run_id = str(args.stage28_run_id).strip()
    stage28_stdout = ""
    stage28_stderr = ""
    if not stage28_run_id:
        stage28_run_id, stage28_stdout, stage28_stderr = _run_stage28(
            config=args.config,
            seed=int(args.seed),
            mode=str(args.mode),
            budget_small=bool(args.budget_small),
            runs_dir=Path(args.runs_dir),
            docs_dir=Path(args.docs_dir),
        )

    stage28_dir = Path(args.runs_dir) / stage28_run_id / "stage28"
    trace = _load_csv(stage28_dir / "policy_trace.csv")
    shadow = _load_csv(stage28_dir / "shadow_live_rejects.csv")
    finalists = _load_csv(stage28_dir / "finalists_stageC.csv")
    summary = _load_json(stage28_dir / "summary.json")
    live_trade_count = float((((summary.get("policy_metrics", {}) or {}).get("live", {}) or {}).get("trade_count", 0.0)))

    context_quality = (
        finalists.groupby("context", dropna=False)["exp_lcb"].max().to_dict()
        if not finalists.empty and {"context", "exp_lcb"}.issubset(set(finalists.columns))
        else {}
    )
    calibration = calibrate_thresholds(
        trace_df=trace,
        context_quality={str(k): float(v) for k, v in context_quality.items()},
        cfg=activation_cfg,
    )
    strict = compute_reject_chain_metrics(
        trace_df=trace,
        shadow_df=shadow,
        finalists_df=finalists,
        threshold=0.0,
        quality_floor=0.0,
        final_trade_count=live_trade_count,
    )
    hunt = compute_reject_chain_metrics(
        trace_df=trace,
        shadow_df=shadow,
        finalists_df=finalists,
        threshold=float(calibration.get("chosen_threshold", 0.0)),
        quality_floor=float(activation_cfg.min_quality_floor),
        final_trade_count=live_trade_count,
    )

    payload = {
        "stage": "37.1",
        "seed": int(args.seed),
        "mode": str(args.mode),
        "budget_small": bool(args.budget_small),
        "stage28_run_id": stage28_run_id,
        "strict": strict,
        "hunt": hunt,
        "threshold_curve": calibration.get("curves", []),
        "chosen_threshold": float(calibration.get("chosen_threshold", 0.0)),
        "main_killer": _main_gate_killer(dict(strict.get("overall", {}))),
        "stage28_stdout_tail": "\n".join(stage28_stdout.splitlines()[-12:]),
        "stage28_stderr_tail": "\n".join(stage28_stderr.splitlines()[-12:]),
    }

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / "stage37_activation_hunt_report.md"
    report_json = docs_dir / "stage37_activation_hunt_summary.json"
    report_md.write_text(_render_markdown(payload), encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    out_dir = stage28_dir / "stage37"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "activation_hunt.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"stage28_run_id: {stage28_run_id}")
    print(f"chosen_threshold: {float(payload['chosen_threshold']):.6f}")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")


if __name__ == "__main__":
    main()
