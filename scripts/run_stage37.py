"""Stage-37 orchestrator: activation hunt, derivatives expansion, self-learning, and reruns."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-37 end-to-end")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="both", choices=["both", "upgraded_only"])
    parser.add_argument("--budget-small", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--resume-from-docs", action="store_true")
    parser.add_argument("--baseline-run-id", type=str, default="")
    parser.add_argument("--upgraded-run-id", type=str, default="")
    parser.add_argument("--skip-seed-reruns", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or "")


def _run_checked(cmd: list[str], *, label: str) -> tuple[str, str]:
    code, out, err = _run(cmd)
    if code != 0:
        tail = "\n".join((out + "\n" + err).splitlines()[-60:])
        raise RuntimeError(f"{label} failed (exit={code})\n{tail}")
    return out, err


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _parse_run_id(stdout: str) -> str:
    match = re.search(r"^run_id:\s*(\S+)\s*$", str(stdout), flags=re.MULTILINE)
    return str(match.group(1)) if match else ""


def _build_upgraded_config(*, base_config: Path, out_path: Path) -> Path:
    payload = yaml.safe_load(Path(base_config).read_text(encoding="utf-8")) or {}
    data = payload.setdefault("data", {})
    futures = data.setdefault("futures_extras", {})
    oi = futures.setdefault("open_interest", {})
    data["include_futures_extras"] = True
    oi["short_horizon_only"] = True
    oi["short_horizon_max"] = "30m"
    eval_cfg = payload.setdefault("evaluation", {})
    stage37 = eval_cfg.setdefault("stage37", {})
    stage37["enabled"] = True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out_path


def _extract_stage28_metrics(run_dir: Path) -> dict[str, Any]:
    summary = _load_json(run_dir / "summary.json")
    trace = pd.read_csv(run_dir / "policy_trace.csv") if (run_dir / "policy_trace.csv").exists() else pd.DataFrame()
    live = dict((summary.get("policy_metrics", {}) or {}).get("live", {})
)
    research = dict((summary.get("policy_metrics", {}) or {}).get("research", {})
)
    raw_signal_count = int((pd.to_numeric(trace.get("final_signal", 0), errors="coerce").fillna(0).astype(int) != 0).sum()) if not trace.empty else 0
    trade_count = float(live.get("trade_count", 0.0))
    return {
        "run_id": str(summary.get("run_id", "")),
        "summary_hash": str(summary.get("summary_hash", "")),
        "raw_signal_count": int(raw_signal_count),
        "activation_rate": float(trade_count / max(1, raw_signal_count)),
        "trade_count": float(trade_count),
        "research_best_exp_lcb": float(research.get("exp_lcb", 0.0)),
        "live_best_exp_lcb": float(live.get("exp_lcb", 0.0)),
        "wf_executed_pct": float(summary.get("wf_executed_pct", 0.0)),
        "mc_trigger_pct": float(summary.get("mc_trigger_pct", 0.0)),
        "next_bottleneck": str(summary.get("next_bottleneck", "")),
    }


def _render_5seed(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-37 5-Seed Report",
        "",
        f"- promising_from_seed42: `{bool(payload.get('promising_from_seed42', False))}`",
        f"- executed_seed_count: `{int(payload.get('executed_seed_count', 0))}`",
        "",
        "## Seed Metrics",
        "| seed | run_id | activation_rate | trade_count | research_exp_lcb | live_exp_lcb |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("rows", []):
        lines.append(
            "| {seed} | {run_id} | {act:.6f} | {trades:.3f} | {r:.6f} | {l:.6f} |".format(
                seed=int(row.get("seed", 0)),
                run_id=str(row.get("run_id", "")),
                act=float(row.get("activation_rate", 0.0)),
                trades=float(row.get("trade_count", 0.0)),
                r=float(row.get("research_best_exp_lcb", 0.0)),
                l=float(row.get("live_best_exp_lcb", 0.0)),
            )
        )
    dist = dict(payload.get("distribution", {}))
    lines.extend(
        [
            "",
            "## Distribution",
            f"- activation_rate_median: `{float(dist.get('activation_rate_median', 0.0)):.6f}`",
            f"- trade_count_median: `{float(dist.get('trade_count_median', 0.0)):.6f}`",
            f"- live_exp_lcb_median: `{float(dist.get('live_exp_lcb_median', 0.0)):.6f}`",
            f"- live_exp_lcb_worst: `{float(dist.get('live_exp_lcb_worst', 0.0)):.6f}`",
            f"- live_exp_lcb_best: `{float(dist.get('live_exp_lcb_best', 0.0)):.6f}`",
        ]
    )
    if str(payload.get("note", "")).strip():
        lines.extend(["", "## Note", f"- {payload['note']}"])
    return "\n".join(lines).strip() + "\n"


def _master_verdict(*, engine: dict[str, Any], five_seed: dict[str, Any], self_learning: dict[str, Any], derivatives: dict[str, Any]) -> str:
    base = dict(engine.get("baseline", {}))
    up = dict(engine.get("upgraded", {}))
    delta = dict(engine.get("delta", {}))
    if float(delta.get("delta_activation_rate", 0.0)) > 0 and float(up.get("trade_count", 0.0)) > float(base.get("trade_count", 0.0)):
        return "ZERO_TRADE_BOTTLENECK_REDUCED"
    if bool(derivatives.get("funding_available", False)) and float(up.get("live_best_exp_lcb", 0.0)) <= 0.0:
        return "DATA_IMPROVED_BUT_NO_EDGE"
    if int(self_learning.get("new_rows_added", 0)) > 0 and float(delta.get("delta_activation_rate", 0.0)) <= 0.0:
        return "SELF_LEARNING_IMPROVED_BUT_SIGNAL_WEAK"
    if int(five_seed.get("executed_seed_count", 0)) <= 1 and not bool(engine.get("promising", False)):
        return "NO_MEANINGFUL_PROGRESS"
    return "PARTIAL_PROGRESS_NEEDS_MORE_FEATURES"


def _render_master(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-37 Master Report",
        "",
        "## Changes",
        "- Added activation-hunt reject-chain diagnostics with threshold sensitivity.",
        "- Expanded free derivatives feature families (funding/taker/long-short) and OI short-horizon mode.",
        "- Added failure-aware self-learning registry and deterministic elite/pruning logic.",
        "",
        "## Data Families",
        f"- funding available: `{payload.get('derivatives', {}).get('funding_available', False)}`",
        f"- taker available: `{payload.get('derivatives', {}).get('taker_available', False)}`",
        f"- long_short available: `{payload.get('derivatives', {}).get('long_short_available', False)}`",
        f"- oi_short_only_mode: `{payload.get('derivatives', {}).get('oi_short_only_mode', False)}`",
        "",
        "## Engine",
        f"- baseline_run_id: `{payload.get('engine', {}).get('baseline', {}).get('run_id', '')}`",
        f"- upgraded_run_id: `{payload.get('engine', {}).get('upgraded', {}).get('run_id', '')}`",
        f"- delta_activation_rate: `{float(payload.get('engine', {}).get('delta', {}).get('delta_activation_rate', 0.0)):.6f}`",
        f"- delta_trade_count: `{float(payload.get('engine', {}).get('delta', {}).get('delta_trade_count', 0.0)):.6f}`",
        f"- delta_live_exp_lcb: `{float(payload.get('engine', {}).get('delta', {}).get('delta_live_best_exp_lcb', 0.0)):.6f}`",
        "",
        "## Stability",
        f"- executed_seed_count: `{int(payload.get('five_seed', {}).get('executed_seed_count', 0))}`",
        f"- note: `{payload.get('five_seed', {}).get('note', '')}`",
        "",
        "## Verdict",
        f"- verdict: `{payload.get('verdict', '')}`",
        f"- biggest_remaining_bottleneck: `{payload.get('biggest_remaining_bottleneck', '')}`",
        f"- next_cheapest_action: `{payload.get('next_cheapest_action', '')}`",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    stage37_run_id = f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'cfg': str(args.config)}, length=12)}_stage37"
    stage37_dir = Path(args.runs_dir) / stage37_run_id / "stage37"
    stage37_dir.mkdir(parents=True, exist_ok=True)

    if not bool(args.resume_from_docs):
        activation_cmd = [
            sys.executable,
            "scripts/stage37_activation_hunt.py",
            "--config",
            str(args.config),
            "--seed",
            str(int(args.seed)),
            "--mode",
            "both",
            "--runs-dir",
            str(args.runs_dir),
            "--docs-dir",
            str(docs_dir),
        ]
        if bool(args.budget_small):
            activation_cmd.append("--budget-small")
        _run_checked(activation_cmd, label="stage37_activation_hunt")

        derivatives_cmd = [
            sys.executable,
            "scripts/stage37_build_derivatives_features.py",
            "--config",
            str(args.config),
            "--data-dir",
            "data/canonical",
            "--derived-dir",
            "data/derived",
            "--docs-dir",
            str(docs_dir),
        ]
        _run_checked(derivatives_cmd, label="stage37_build_derivatives_features")

        self_learning_cmd = [
            sys.executable,
            "scripts/stage37_self_learning_rerun.py",
            "--config",
            str(args.config),
            "--seed",
            str(int(args.seed)),
            "--runs-dir",
            str(args.runs_dir),
            "--docs-dir",
            str(docs_dir),
        ]
        _run_checked(self_learning_cmd, label="stage37_self_learning_rerun")

    activation_summary = _load_json(docs_dir / "stage37_activation_hunt_summary.json")
    derivatives_summary = _load_json(docs_dir / "stage37_derivatives_expansion_summary.json")
    self_learning_summary = _load_json(docs_dir / "stage37_self_learning_upgrade_summary.json")

    upgraded_cfg = _build_upgraded_config(base_config=Path(args.config), out_path=stage37_dir / "stage37_upgraded.yaml")
    compare_cmd = [
        sys.executable,
        "scripts/stage37_compare_runs.py",
        "--baseline-config",
        str(args.config),
        "--upgraded-config",
        str(upgraded_cfg),
        "--seed",
        str(int(args.seed)),
        "--runs-dir",
        str(args.runs_dir),
        "--docs-dir",
        str(docs_dir),
    ]
    if str(args.baseline_run_id).strip():
        compare_cmd.extend(["--baseline-run-id", str(args.baseline_run_id).strip()])
    if str(args.upgraded_run_id).strip():
        compare_cmd.extend(["--upgraded-run-id", str(args.upgraded_run_id).strip()])
    if bool(args.budget_small):
        compare_cmd.append("--budget-small")
    if not (bool(args.resume_from_docs) and not str(args.baseline_run_id).strip() and not str(args.upgraded_run_id).strip() and (docs_dir / "stage37_engine_summary.json").exists()):
        _run_checked(compare_cmd, label="stage37_compare_runs")
    engine_summary = _load_json(docs_dir / "stage37_engine_summary.json")
    promising = bool(engine_summary.get("promising", False))

    seed_rows: list[dict[str, Any]] = []
    seed_list = [43, 44, 45, 46, 47] if (promising and not bool(args.skip_seed_reruns)) else []
    note = ""
    if not promising:
        note = "Not promising on seed-42; skipped extra seeds to avoid waste."
    elif bool(args.skip_seed_reruns):
        note = "Promising but seed reruns explicitly skipped."
    for seed in seed_list:
        cmd = [
            sys.executable,
            "scripts/run_stage28.py",
            "--config",
            str(upgraded_cfg),
            "--seed",
            str(int(seed)),
            "--mode",
            "both",
            "--runs-dir",
            str(args.runs_dir),
            "--docs-dir",
            str(docs_dir),
        ]
        if bool(args.budget_small):
            cmd.append("--budget-small")
        out, _ = _run_checked(cmd, label=f"stage28_upgraded_seed_{seed}")
        run_id = _parse_run_id(out)
        if not run_id:
            continue
        metrics = _extract_stage28_metrics(Path(args.runs_dir) / run_id / "stage28")
        seed_rows.append({"seed": int(seed), **metrics})

    dist = {
        "activation_rate_median": float(pd.Series([row.get("activation_rate", 0.0) for row in seed_rows], dtype=float).median()) if seed_rows else 0.0,
        "trade_count_median": float(pd.Series([row.get("trade_count", 0.0) for row in seed_rows], dtype=float).median()) if seed_rows else 0.0,
        "live_exp_lcb_median": float(pd.Series([row.get("live_best_exp_lcb", 0.0) for row in seed_rows], dtype=float).median()) if seed_rows else 0.0,
        "live_exp_lcb_worst": float(pd.Series([row.get("live_best_exp_lcb", 0.0) for row in seed_rows], dtype=float).min()) if seed_rows else 0.0,
        "live_exp_lcb_best": float(pd.Series([row.get("live_best_exp_lcb", 0.0) for row in seed_rows], dtype=float).max()) if seed_rows else 0.0,
    }
    five_seed_payload = {
        "stage": "37.5",
        "promising_from_seed42": bool(promising),
        "executed_seed_count": int(len(seed_rows)),
        "rows": seed_rows,
        "distribution": dist,
        "note": note,
    }
    (docs_dir / "stage37_5seed_summary.json").write_text(json.dumps(five_seed_payload, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage37_5seed_report.md").write_text(_render_5seed(five_seed_payload), encoding="utf-8")

    # Determinism check: rerun seed-42 upgraded and compare summary hash.
    deterministic = False
    if not bool(args.skip_seed_reruns):
        det_cmd = [
            sys.executable,
            "scripts/run_stage28.py",
            "--config",
            str(upgraded_cfg),
            "--seed",
            str(int(args.seed)),
            "--mode",
            "both",
            "--runs-dir",
            str(args.runs_dir),
            "--docs-dir",
            str(docs_dir),
        ]
        if bool(args.budget_small):
            det_cmd.append("--budget-small")
        det_out, _ = _run_checked(det_cmd, label="stage28_upgraded_seed42_rerun")
        det_run_id = _parse_run_id(det_out)
        det_metrics = _extract_stage28_metrics(Path(args.runs_dir) / det_run_id / "stage28") if det_run_id else {}
        deterministic = bool(
            det_metrics
            and str(det_metrics.get("summary_hash", "")) == str((engine_summary.get("upgraded", {}) or {}).get("summary_hash", ""))
        )

    biggest_bottleneck = str((engine_summary.get("upgraded", {}) or {}).get("next_bottleneck", "signal_quality"))
    next_action = "Tune Stage-28/37 activation and cost-gate settings per family using stage37_activation_hunt_report.md"
    master_payload = {
        "stage": "37.6",
        "run_id": stage37_run_id,
        "seed": int(args.seed),
        "activation": activation_summary,
        "derivatives": derivatives_summary,
        "self_learning": self_learning_summary,
        "engine": engine_summary,
        "five_seed": five_seed_payload,
        "deterministic_seed42_rerun": bool(deterministic),
    }
    master_payload["verdict"] = _master_verdict(
        engine=engine_summary,
        five_seed=five_seed_payload,
        self_learning=self_learning_summary,
        derivatives=derivatives_summary,
    )
    master_payload["biggest_remaining_bottleneck"] = biggest_bottleneck
    master_payload["next_cheapest_action"] = next_action

    (docs_dir / "stage37_master_summary.json").write_text(json.dumps(master_payload, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage37_master_report.md").write_text(_render_master(master_payload), encoding="utf-8")

    print(f"stage37_run_id: {stage37_run_id}")
    print(f"activation_report: {docs_dir / 'stage37_activation_hunt_report.md'}")
    print(f"derivatives_report: {docs_dir / 'stage37_derivatives_expansion_report.md'}")
    print(f"self_learning_report: {docs_dir / 'stage37_self_learning_upgrade.md'}")
    print(f"engine_report: {docs_dir / 'stage37_engine_report.md'}")
    print(f"seed_report: {docs_dir / 'stage37_5seed_report.md'}")
    print(f"master_report: {docs_dir / 'stage37_master_report.md'}")
    print(f"verdict: {master_payload['verdict']}")
    print(f"deterministic_seed42_rerun: {deterministic}")


if __name__ == "__main__":
    main()
