"""Stage-43 performance instrumentation and 5-seed validation."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.store import build_data_store
from buffmini.stage39.signal_generation import build_layered_candidates
from buffmini.stage40.objective import (
    TradabilityConfig,
    compute_tradability_labels,
    route_two_stage_objective_with_trace,
)
from buffmini.stage41.contribution import (
    compute_family_contribution_metrics,
    oi_short_only_runtime_guard,
)
from buffmini.stage43.reporting import (
    REQUIRED_PHASE_KEYS,
    render_stage43_five_seed_report,
    render_stage43_performance_report,
    validate_stage43_5seed_summary,
    validate_stage43_performance_summary,
)
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-43 performance instrumentation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--budget-small", action="store_true")
    parser.add_argument("--baseline-run-id", type=str, default="")
    parser.add_argument("--upgraded-run-id", type=str, default="")
    parser.add_argument("--skip-seed-validation", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_csv(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _parse_run_id(stdout: str) -> str:
    match = re.search(r"^run_id:\s*(\S+)\s*$", stdout, flags=re.MULTILINE)
    return str(match.group(1)) if match else ""


def _run_stage28(
    *,
    config: Path,
    seed: int,
    runs_dir: Path,
    docs_dir: Path,
    budget_small: bool,
) -> tuple[str, float]:
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
    started = time.perf_counter()
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    elapsed = float(time.perf_counter() - started)
    stdout = str(proc.stdout or "")
    stderr = str(proc.stderr or "")
    if int(proc.returncode) != 0:
        tail = "\n".join((stdout + "\n" + stderr).splitlines()[-80:])
        raise RuntimeError(f"run_stage28 failed (exit={proc.returncode})\n{tail}")
    run_id = _parse_run_id(stdout)
    if not run_id:
        raise RuntimeError("run_stage28 did not emit run_id")
    return run_id, elapsed


def _extract_stage28_metrics(run_dir: Path) -> dict[str, Any]:
    summary = _load_json(run_dir / "summary.json")
    trace = _load_csv(run_dir / "policy_trace.csv")
    rejects = _load_csv(run_dir / "shadow_live_rejects.csv")
    live = dict((summary.get("policy_metrics", {}) or {}).get("live", {}))
    research = dict((summary.get("policy_metrics", {}) or {}).get("research", {}))
    final_signal = pd.to_numeric(trace.get("final_signal", 0), errors="coerce").fillna(0).astype(int)
    raw_signal_count = int((final_signal != 0).sum()) if not trace.empty else 0
    reject_map = (
        rejects["reason"].astype(str).value_counts(dropna=False).head(8).to_dict()
        if ("reason" in rejects.columns and not rejects.empty)
        else {}
    )
    return {
        "run_id": str(summary.get("run_id", "")),
        "summary_hash": str(summary.get("summary_hash", "")),
        "raw_signal_count": int(raw_signal_count),
        "activation_rate": float(float(live.get("trade_count", 0.0)) / max(1, raw_signal_count)),
        "trade_count": float(live.get("trade_count", 0.0)),
        "research_best_exp_lcb": float(research.get("exp_lcb", 0.0)),
        "live_best_exp_lcb": float(live.get("exp_lcb", 0.0)),
        "wf_executed_pct": float(summary.get("wf_executed_pct", 0.0)),
        "mc_trigger_pct": float(summary.get("mc_trigger_pct", 0.0)),
        "runtime_seconds": float(summary.get("runtime_seconds", 0.0)),
        "top_reject_reasons": reject_map,
        "next_bottleneck": str(summary.get("next_bottleneck", "")),
        "verdict": str(summary.get("verdict", "")),
    }


def _delta_metrics(base: dict[str, Any], up: dict[str, Any]) -> dict[str, float]:
    keys = (
        "raw_signal_count",
        "activation_rate",
        "trade_count",
        "research_best_exp_lcb",
        "live_best_exp_lcb",
        "runtime_seconds",
    )
    return {f"delta_{key}": float(up.get(key, 0.0)) - float(base.get(key, 0.0)) for key in keys}


def _promising(base: dict[str, Any], up: dict[str, Any]) -> bool:
    if float(up.get("raw_signal_count", 0.0)) > float(base.get("raw_signal_count", 0.0)):
        return True
    if float(up.get("activation_rate", 0.0)) > float(base.get("activation_rate", 0.0)) + 1e-12:
        return True
    if float(up.get("trade_count", 0.0)) > float(base.get("trade_count", 0.0)):
        return True
    if float(up.get("research_best_exp_lcb", 0.0)) > float(base.get("research_best_exp_lcb", 0.0)) + 1e-12:
        return True
    if float(up.get("live_best_exp_lcb", 0.0)) > float(base.get("live_best_exp_lcb", 0.0)) + 1e-12:
        return True
    return False


def _build_upgraded_config(*, base_config: Path, out_path: Path) -> Path:
    payload = yaml.safe_load(Path(base_config).read_text(encoding="utf-8")) or {}
    data = payload.setdefault("data", {})
    futures = data.setdefault("futures_extras", {})
    oi = futures.setdefault("open_interest", {})
    data["include_futures_extras"] = True
    oi["short_horizon_only"] = True
    oi["short_horizon_max"] = "30m"
    eval_cfg = payload.setdefault("evaluation", {})
    stage39_43 = eval_cfg.setdefault("stage39_43", {})
    stage39_43["enabled"] = True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out_path


def _safe_ohlcv(config: dict[str, Any]) -> pd.DataFrame:
    symbols = list(((config.get("universe", {}) or {}).get("symbols", ["BTC/USDT"])))
    symbol = str(symbols[0]) if symbols else "BTC/USDT"
    timeframe = str((config.get("universe", {}) or {}).get("operational_timeframe", "1h"))
    store = build_data_store(
        backend=str((config.get("data", {}) or {}).get("backend", "parquet")),
        data_dir=RAW_DATA_DIR,
        base_timeframe=str((config.get("universe", {}) or {}).get("base_timeframe", "1m")),
        resample_source=str((config.get("data", {}) or {}).get("resample_source", "direct")),
        derived_dir=Path("data") / "derived",
        partial_last_bucket=bool((config.get("data", {}) or {}).get("partial_last_bucket", False)),
    )
    bars = store.load_ohlcv(symbol=symbol, timeframe=timeframe).tail(600).reset_index(drop=True)
    if not bars.empty:
        return bars[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    idx = pd.date_range("2025-01-01T00:00:00Z", periods=600, freq="1h", tz="UTC")
    base = 100.0 + np.cumsum(np.sin(np.arange(600) / 24.0) * 0.1)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": base,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base + 0.05,
            "volume": 900.0,
        }
    )


def _seed_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    raw = pd.Series([float(row.get("raw_candidate_count", 0.0)) for row in rows], dtype=float)
    act = pd.Series([float(row.get("activation_rate", 0.0)) for row in rows], dtype=float)
    trd = pd.Series([float(row.get("trade_count", 0.0)) for row in rows], dtype=float)
    live = pd.Series([float(row.get("live_best_exp_lcb", 0.0)) for row in rows], dtype=float)
    if raw.empty:
        return {
            "raw_candidate_count_median": 0.0,
            "activation_rate_median": 0.0,
            "trade_count_median": 0.0,
            "live_exp_lcb_median": 0.0,
            "live_exp_lcb_worst": 0.0,
            "live_exp_lcb_best": 0.0,
        }
    return {
        "raw_candidate_count_median": float(raw.median()),
        "activation_rate_median": float(act.median()),
        "trade_count_median": float(trd.median()),
        "live_exp_lcb_median": float(live.median()),
        "live_exp_lcb_worst": float(live.min()),
        "live_exp_lcb_best": float(live.max()),
    }


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)

    phase_runtime: dict[str, float] = {key: 0.0 for key in REQUIRED_PHASE_KEYS}
    generated_stage28_elapsed: dict[str, float] = {}

    t0 = time.perf_counter()
    config = load_config(Path(args.config))
    phase_runtime["config_load"] = float(time.perf_counter() - t0)

    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    stage40 = _load_json(docs_dir / "stage40_tradability_objective_summary.json")
    stage41 = _load_json(docs_dir / "stage41_derivatives_completion_summary.json")
    stage42 = _load_json(docs_dir / "stage42_self_learning_2_summary.json")
    stage37_engine = _load_json(docs_dir / "stage37_engine_summary.json")
    stage37_seeds = _load_json(docs_dir / "stage37_5seed_summary.json")

    stage43_run_id = f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'config': str(args.config)}, length=12)}_stage43"
    stage43_dir = runs_dir / stage43_run_id / "stage43"
    stage43_dir.mkdir(parents=True, exist_ok=True)

    baseline_run_id = str(args.baseline_run_id).strip() or str((stage37_engine.get("baseline", {}) or {}).get("run_id", "")).strip()
    upgraded_run_id = str(args.upgraded_run_id).strip() or str((stage37_engine.get("upgraded", {}) or {}).get("run_id", "")).strip()

    upgraded_config_path = _build_upgraded_config(
        base_config=Path(args.config),
        out_path=stage43_dir / "stage43_upgraded.yaml",
    )

    if not baseline_run_id or not (runs_dir / baseline_run_id / "stage28").exists():
        baseline_run_id, elapsed = _run_stage28(
            config=Path(args.config),
            seed=int(args.seed),
            runs_dir=runs_dir,
            docs_dir=docs_dir,
            budget_small=bool(args.budget_small),
        )
        generated_stage28_elapsed["baseline"] = float(elapsed)
    if not upgraded_run_id or not (runs_dir / upgraded_run_id / "stage28").exists():
        upgraded_run_id, elapsed = _run_stage28(
            config=upgraded_config_path,
            seed=int(args.seed),
            runs_dir=runs_dir,
            docs_dir=docs_dir,
            budget_small=bool(args.budget_small),
        )
        generated_stage28_elapsed["upgraded"] = float(elapsed)

    baseline_metrics = _extract_stage28_metrics(runs_dir / baseline_run_id / "stage28")
    upgraded_metrics = _extract_stage28_metrics(runs_dir / upgraded_run_id / "stage28")
    if float(upgraded_metrics.get("runtime_seconds", 0.0)) <= 0.0:
        upgraded_metrics["runtime_seconds"] = float(generated_stage28_elapsed.get("upgraded", 0.0))
    if float(baseline_metrics.get("runtime_seconds", 0.0)) <= 0.0:
        baseline_metrics["runtime_seconds"] = float(generated_stage28_elapsed.get("baseline", 0.0))

    # Stage-43 phase instrumentation using deterministic local pipeline operations.
    stage28_reference_id = str(stage39.get("stage28_run_id", "")).strip() or str(upgraded_run_id)
    finalists_path = runs_dir / stage28_reference_id / "stage28" / "finalists_stageC.csv"

    t0 = time.perf_counter()
    finalists = _load_csv(finalists_path)
    bars = _safe_ohlcv(config)
    phase_runtime["data_load"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    layered = build_layered_candidates(finalists, seed=int(args.seed))
    phase_runtime["candidate_generation"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    oi_cfg = dict(stage41.get("oi_runtime_guard", {}))
    _ = oi_short_only_runtime_guard(
        timeframe=str(oi_cfg.get("timeframe", "1h")),
        short_only_enabled=bool(stage41.get("oi_short_only_mode_enabled", True)),
        short_horizon_max=str(oi_cfg.get("short_horizon_max", "30m")),
    )
    phase_runtime["extras_alignment"] = float(time.perf_counter() - t0)

    labels = compute_tradability_labels(
        bars,
        cfg=TradabilityConfig(
            horizon_bars=12,
            tp_pct=0.004,
            sl_pct=0.003,
            round_trip_cost_pct=float((config.get("costs", {}) or {}).get("round_trip_cost_pct", 0.1)) / 100.0,
            max_adverse_excursion_pct=0.004,
            stage_a_threshold=0.35,
            stage_b_threshold=0.0,
        ),
    )
    routed = route_two_stage_objective_with_trace(
        layered.layer_c,
        labels=labels,
        cfg=TradabilityConfig(
            horizon_bars=12,
            tp_pct=0.004,
            sl_pct=0.003,
            round_trip_cost_pct=float((config.get("costs", {}) or {}).get("round_trip_cost_pct", 0.1)) / 100.0,
            max_adverse_excursion_pct=0.004,
            stage_a_threshold=0.35,
            stage_b_threshold=0.0,
        ),
    )
    phase_runtime["stage_a_objective"] = float((routed.get("timings", {}) or {}).get("stage_a_seconds", 0.0))
    phase_runtime["stage_b_objective"] = float((routed.get("timings", {}) or {}).get("stage_b_seconds", 0.0))

    t0 = time.perf_counter()
    _ = compute_family_contribution_metrics(
        layer_a=layered.layer_a,
        layer_c=layered.layer_c,
        stage_a_survivors=pd.DataFrame(routed.get("stage_a_survivors", pd.DataFrame())),
        stage_b_survivors=pd.DataFrame(routed.get("stage_b_survivors", pd.DataFrame())),
        families=[str(row.get("family", "")) for row in stage41.get("family_contributions", []) if str(row.get("family", "")).strip()],
    )
    phase_runtime["feature_generation"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    policy_trace = _load_csv(runs_dir / upgraded_run_id / "stage28" / "policy_trace.csv")
    _composer_rows = int(
        (
            pd.to_numeric(policy_trace.get("final_signal", 0), errors="coerce")
            .fillna(0)
            .astype(int)
            != 0
        ).sum()
    ) if not policy_trace.empty else 0
    phase_runtime["composer_policy_build"] = float(time.perf_counter() - t0)

    # run_stage28 is monolithic; replay/WF/MC do not currently expose split timers.
    phase_runtime["replay_backtest"] = float(upgraded_metrics.get("runtime_seconds", 0.0))
    phase_runtime["walkforward"] = 0.0
    phase_runtime["monte_carlo"] = 0.0

    delta = _delta_metrics(baseline_metrics, upgraded_metrics)
    performance_payload = {
        "stage": "43.3",
        "seed": int(args.seed),
        "baseline": baseline_metrics,
        "upgraded": upgraded_metrics,
        "delta": delta,
        "promising": bool(_promising(baseline_metrics, upgraded_metrics)),
        "phase_runtime_seconds": {k: float(phase_runtime.get(k, 0.0)) for k in REQUIRED_PHASE_KEYS},
        "slowest_phase": max(REQUIRED_PHASE_KEYS, key=lambda key: float(phase_runtime.get(key, 0.0))),
        "budget_mode": "small" if bool(args.budget_small) else "standard",
        "runtime_notes": "run_stage28 exposes total runtime only; walkforward/monte_carlo remain embedded in replay_backtest runtime.",
        "stage39_counts": {
            "raw_candidate_count": int(stage39.get("raw_candidate_count", 0)),
            "shortlisted_count": int(stage39.get("shortlisted_count", 0)),
        },
        "stage40_counts": dict(stage40.get("counts", {})),
        "stage41_top_family": str((stage41.get("family_contributions", [{}]) or [{}])[0].get("family", "")),
        "stage42_registry_rows_v2": int(stage42.get("registry_rows_v2", 0)),
    }

    report_started = time.perf_counter()
    performance_payload["summary_hash"] = stable_hash(
        {
            "stage": performance_payload["stage"],
            "seed": performance_payload["seed"],
            "baseline": performance_payload["baseline"],
            "upgraded": performance_payload["upgraded"],
            "delta": performance_payload["delta"],
            "promising": performance_payload["promising"],
            "phase_runtime_seconds": performance_payload["phase_runtime_seconds"],
            "stage39_counts": performance_payload["stage39_counts"],
            "stage40_counts": performance_payload["stage40_counts"],
            "stage42_registry_rows_v2": performance_payload["stage42_registry_rows_v2"],
        },
        length=16,
    )
    validate_stage43_performance_summary(performance_payload)

    perf_report_path = docs_dir / "stage43_performance_report.md"
    perf_summary_path = docs_dir / "stage43_performance_summary.json"
    perf_report_path.write_text(render_stage43_performance_report(performance_payload), encoding="utf-8")
    perf_summary_path.write_text(json.dumps(performance_payload, indent=2, allow_nan=False), encoding="utf-8")

    # 5-seed validation
    skip_reason = ""
    seed_rows: list[dict[str, Any]] = []
    if bool(args.skip_seed_validation):
        skip_reason = "explicit_skip_flag"
    elif not bool(performance_payload.get("promising", False)) and int(upgraded_metrics.get("raw_signal_count", 0)) <= 0:
        skip_reason = "upgraded_run_dead_zero_signal_no_improvement"

    if not skip_reason:
        reusable_rows = [row for row in stage37_seeds.get("rows", []) if isinstance(row, dict)]
        reusable_map = {int(row.get("seed", -1)): dict(row) for row in reusable_rows if int(row.get("seed", -1)) in {43, 44, 45, 46, 47}}
        for seed in (43, 44, 45, 46, 47):
            row = reusable_map.get(int(seed))
            if row and str(row.get("run_id", "")).strip() and (runs_dir / str(row.get("run_id")) / "stage28").exists():
                metrics = _extract_stage28_metrics(runs_dir / str(row["run_id"]) / "stage28")
            else:
                run_id, _ = _run_stage28(
                    config=upgraded_config_path,
                    seed=int(seed),
                    runs_dir=runs_dir,
                    docs_dir=docs_dir,
                    budget_small=bool(args.budget_small),
                )
                metrics = _extract_stage28_metrics(runs_dir / run_id / "stage28")
            seed_rows.append(
                {
                    "seed": int(seed),
                    "run_id": str(metrics.get("run_id", "")),
                    "raw_candidate_count": int(stage39.get("raw_candidate_count", 0)),
                    "raw_signal_count": int(metrics.get("raw_signal_count", 0)),
                    "activation_rate": float(metrics.get("activation_rate", 0.0)),
                    "trade_count": float(metrics.get("trade_count", 0.0)),
                    "live_best_exp_lcb": float(metrics.get("live_best_exp_lcb", 0.0)),
                }
            )

    five_seed_payload = {
        "stage": "43.4",
        "seed": int(args.seed),
        "upgraded_reference_run_id": str(upgraded_metrics.get("run_id", "")),
        "executed_seed_count": int(len(seed_rows)),
        "skipped": bool(skip_reason),
        "skip_reason": str(skip_reason),
        "note": (
            "Skipped 5-seed validation because upgraded seed-42 remained fully dead (zero raw signals and zero trades)."
            if skip_reason == "upgraded_run_dead_zero_signal_no_improvement"
            else ("Skipped due to explicit flag." if skip_reason else "Executed all five validation seeds.")
        ),
        "rows": seed_rows,
        "distribution": _seed_distribution(seed_rows),
    }
    five_seed_payload["summary_hash"] = stable_hash(
        {
            "stage": five_seed_payload["stage"],
            "seed": five_seed_payload["seed"],
            "upgraded_reference_run_id": five_seed_payload["upgraded_reference_run_id"],
            "executed_seed_count": five_seed_payload["executed_seed_count"],
            "skipped": five_seed_payload["skipped"],
            "skip_reason": five_seed_payload["skip_reason"],
            "rows": five_seed_payload["rows"],
            "distribution": five_seed_payload["distribution"],
        },
        length=16,
    )
    validate_stage43_5seed_summary(five_seed_payload)

    seed_report_path = docs_dir / "stage43_5seed_report.md"
    seed_summary_path = docs_dir / "stage43_5seed_summary.json"
    seed_report_path.write_text(render_stage43_five_seed_report(five_seed_payload), encoding="utf-8")
    seed_summary_path.write_text(json.dumps(five_seed_payload, indent=2, allow_nan=False), encoding="utf-8")

    phase_runtime["report_generation"] = float(time.perf_counter() - report_started)
    performance_payload["phase_runtime_seconds"]["report_generation"] = float(phase_runtime["report_generation"])
    performance_payload["slowest_phase"] = max(
        REQUIRED_PHASE_KEYS,
        key=lambda key: float(performance_payload["phase_runtime_seconds"].get(key, 0.0)),
    )
    performance_payload["summary_hash"] = stable_hash(
        {
            "stage": performance_payload["stage"],
            "seed": performance_payload["seed"],
            "baseline": performance_payload["baseline"],
            "upgraded": performance_payload["upgraded"],
            "delta": performance_payload["delta"],
            "promising": performance_payload["promising"],
            "phase_runtime_seconds": performance_payload["phase_runtime_seconds"],
            "stage39_counts": performance_payload["stage39_counts"],
            "stage40_counts": performance_payload["stage40_counts"],
            "stage42_registry_rows_v2": performance_payload["stage42_registry_rows_v2"],
            "five_seed_hash": five_seed_payload["summary_hash"],
        },
        length=16,
    )
    validate_stage43_performance_summary(performance_payload)
    perf_report_path.write_text(render_stage43_performance_report(performance_payload), encoding="utf-8")
    perf_summary_path.write_text(json.dumps(performance_payload, indent=2, allow_nan=False), encoding="utf-8")

    (stage43_dir / "performance_summary.json").write_text(
        json.dumps(performance_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (stage43_dir / "five_seed_summary.json").write_text(
        json.dumps(five_seed_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"stage43_run_id: {stage43_run_id}")
    print(f"baseline_run_id: {baseline_run_id}")
    print(f"upgraded_run_id: {upgraded_run_id}")
    print(f"performance_summary_hash: {performance_payload['summary_hash']}")
    print(f"five_seed_summary_hash: {five_seed_payload['summary_hash']}")
    print(f"performance_summary: {perf_summary_path}")
    print(f"five_seed_summary: {seed_summary_path}")


if __name__ == "__main__":
    main()

