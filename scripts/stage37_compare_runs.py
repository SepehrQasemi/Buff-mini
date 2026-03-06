"""Stage-37.4 baseline-vs-upgraded Stage-28 comparison."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage37.reporting import validate_stage37_engine_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and upgraded Stage-28 runs for Stage-37")
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--upgraded-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--budget-small", action="store_true")
    return parser.parse_args()


def _run_stage28(*, config: Path, seed: int, runs_dir: Path, docs_dir: Path, budget_small: bool) -> tuple[str, str, str]:
    cmd = [
        sys.executable,
        "scripts/run_stage28.py",
        "--config",
        str(config),
        "--seed",
        str(int(seed)),
        "--mode",
        "both",
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
        tail = "\n".join((stdout + "\n" + stderr).splitlines()[-50:])
        raise RuntimeError(f"run_stage28 failed (exit={proc.returncode})\n{tail}")
    match = re.search(r"^run_id:\s*(\S+)\s*$", stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("run_stage28 output missing run_id")
    return str(match.group(1)), stdout, stderr


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _extract_metrics(run_dir: Path) -> dict[str, Any]:
    summary = _load_json(run_dir / "summary.json")
    trace = _load_csv(run_dir / "policy_trace.csv")
    rejects = _load_csv(run_dir / "shadow_live_rejects.csv")
    live = dict((summary.get("policy_metrics", {}) or {}).get("live", {}))
    research = dict((summary.get("policy_metrics", {}) or {}).get("research", {}))
    raw_signal_count = int((pd.to_numeric(trace.get("final_signal", 0), errors="coerce").fillna(0).astype(int) != 0).sum()) if not trace.empty else 0
    final_trade_count = float(live.get("trade_count", 0.0))
    activation_rate = float(final_trade_count / max(1, raw_signal_count))
    top_rejects = rejects["reason"].astype(str).value_counts(dropna=False).head(8).to_dict() if ("reason" in rejects.columns and not rejects.empty) else {}
    return {
        "run_id": str(summary.get("run_id", "")),
        "summary_hash": str(summary.get("summary_hash", "")),
        "wf_executed_pct": float(summary.get("wf_executed_pct", 0.0)),
        "mc_trigger_pct": float(summary.get("mc_trigger_pct", 0.0)),
        "raw_signal_count": raw_signal_count,
        "activation_rate": activation_rate,
        "trade_count": final_trade_count,
        "research_best_exp_lcb": float(research.get("exp_lcb", 0.0)),
        "live_best_exp_lcb": float(live.get("exp_lcb", 0.0)),
        "maxDD": float(live.get("maxDD", 0.0)),
        "top_reject_reasons": top_rejects,
        "verdict": str(summary.get("verdict", "")),
        "next_bottleneck": str(summary.get("next_bottleneck", "")),
    }


def _delta_metrics(base: dict[str, Any], up: dict[str, Any]) -> dict[str, float]:
    keys = [
        "wf_executed_pct",
        "mc_trigger_pct",
        "raw_signal_count",
        "activation_rate",
        "trade_count",
        "research_best_exp_lcb",
        "live_best_exp_lcb",
        "maxDD",
    ]
    return {f"delta_{key}": float(up.get(key, 0.0)) - float(base.get(key, 0.0)) for key in keys}


def _is_promising(base: dict[str, Any], up: dict[str, Any]) -> bool:
    if float(up.get("activation_rate", 0.0)) > float(base.get("activation_rate", 0.0)) + 1e-9:
        return True
    if float(up.get("trade_count", 0.0)) > float(base.get("trade_count", 0.0)):
        return True
    if float(up.get("live_best_exp_lcb", 0.0)) > float(base.get("live_best_exp_lcb", 0.0)) + 1e-9:
        return True
    if float(up.get("research_best_exp_lcb", 0.0)) > float(base.get("research_best_exp_lcb", 0.0)) + 1e-9:
        return True
    return False


def _render_report(payload: dict[str, Any]) -> str:
    baseline = dict(payload.get("baseline", {}))
    upgraded = dict(payload.get("upgraded", {}))
    delta = dict(payload.get("delta", {}))
    lines = [
        "# Stage-37 Engine Report",
        "",
        "## Runs",
        f"- baseline_run_id: `{baseline.get('run_id', '')}`",
        f"- upgraded_run_id: `{upgraded.get('run_id', '')}`",
        "",
        "## Baseline vs Upgraded",
        "| metric | baseline | upgraded | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in (
        "wf_executed_pct",
        "mc_trigger_pct",
        "raw_signal_count",
        "activation_rate",
        "trade_count",
        "research_best_exp_lcb",
        "live_best_exp_lcb",
        "maxDD",
    ):
        lines.append(
            "| {k} | {b:.6f} | {u:.6f} | {d:.6f} |".format(
                k=key,
                b=float(baseline.get(key, 0.0)),
                u=float(upgraded.get(key, 0.0)),
                d=float(delta.get(f"delta_{key}", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## Verdict",
            f"- promising: `{bool(payload.get('promising', False))}`",
            f"- upgraded_verdict: `{upgraded.get('verdict', '')}`",
            f"- upgraded_next_bottleneck: `{upgraded.get('next_bottleneck', '')}`",
            "",
            "## Top Reject Reasons (Upgraded)",
        ]
    )
    reasons = dict(upgraded.get("top_reject_reasons", {}))
    if reasons:
        for key, value in sorted(reasons.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            lines.append(f"- {key}: {int(value)}")
    else:
        lines.append("- none")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    baseline_run_id, _, _ = _run_stage28(
        config=Path(args.baseline_config),
        seed=int(args.seed),
        runs_dir=Path(args.runs_dir),
        docs_dir=Path(args.docs_dir),
        budget_small=bool(args.budget_small),
    )
    upgraded_run_id, _, _ = _run_stage28(
        config=Path(args.upgraded_config),
        seed=int(args.seed),
        runs_dir=Path(args.runs_dir),
        docs_dir=Path(args.docs_dir),
        budget_small=bool(args.budget_small),
    )

    baseline_dir = Path(args.runs_dir) / baseline_run_id / "stage28"
    upgraded_dir = Path(args.runs_dir) / upgraded_run_id / "stage28"
    baseline = _extract_metrics(baseline_dir)
    upgraded = _extract_metrics(upgraded_dir)
    delta = _delta_metrics(baseline, upgraded)
    payload = {
        "stage": "37.4",
        "seed": int(args.seed),
        "baseline": baseline,
        "upgraded": upgraded,
        "delta": delta,
        "promising": bool(_is_promising(baseline, upgraded)),
    }
    validate_stage37_engine_summary(payload)

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / "stage37_engine_report.md"
    report_json = docs_dir / "stage37_engine_summary.json"
    report_md.write_text(_render_report(payload), encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"baseline_run_id: {baseline_run_id}")
    print(f"upgraded_run_id: {upgraded_run_id}")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")


if __name__ == "__main__":
    main()
