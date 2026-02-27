"""Stage-5 pipeline orchestrator.

Runs Stage-1 -> Stage-2 -> Stage-3.3 -> Stage-4 spec (+ optional Stage-4 simulation)
with progress tracking and cancellation support.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, PROJECT_ROOT, RUNS_DIR
from buffmini.data.storage import parquet_path
from buffmini.ui_bundle.builder import build_ui_bundle_from_pipeline
from buffmini.ui.components.run_lock import release_lock
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


class PipelineCancelled(RuntimeError):
    """Raised when a cancel flag is detected during the pipeline."""


@dataclass
class StageResult:
    """Result metadata for one completed stage."""

    name: str
    run_id: str | None
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-5 end-to-end pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--window-months", type=int, choices=[3, 6, 12, 36], default=12)
    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--mode", type=str, choices=["quick", "full"], default="quick")
    parser.add_argument("--execution-mode", type=str, choices=["net", "hedge", "isolated"], default="net")
    parser.add_argument("--fees-round-trip-pct", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-stage4-simulate", type=int, choices=[0, 1], default=0)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()

    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    if not symbols:
        raise ValueError("--symbols must contain at least one symbol")
    if args.timeframe != "1h":
        raise ValueError("Only 1h timeframe is supported")

    signature = stable_hash(
        {
            "symbols": symbols,
            "timeframe": args.timeframe,
            "window_months": args.window_months,
            "mode": args.mode,
            "execution_mode": args.execution_mode,
            "seed": args.seed,
        },
        length=10,
    )
    run_id = args.run_id or f"{utc_now_compact()}_{signature}_stage5_pipeline"
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stage_names = ["data_validate", "stage1", "stage2", "stage3_3", "stage4_spec"]
    if int(args.run_stage4_simulate) == 1:
        stage_names.append("stage4_sim")
    stage_total = len(stage_names)

    stage_results: dict[str, StageResult] = {}
    pipeline_cfg_path = run_dir / "pipeline_config.yaml"
    stage1_run_id: str | None = None
    stage2_run_id: str | None = None
    stage3_run_id: str | None = None
    stage4_run_id: str | None = None
    stage4_sim_run_id: str | None = None
    resolved_candidate_count = 0
    config: dict[str, Any] = {}

    try:
        config = load_config(args.config)
        config, resolved_candidate_count = _prepare_config(config=config, args=args, symbols=symbols)
        pipeline_cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        _run_data_validate(
            symbols=symbols,
            timeframe=args.timeframe,
            run_dir=run_dir,
            started_at=started_at,
            stage_idx=1,
            stage_total=stage_total,
        )

        stage1_run_id = f"{run_id}_stage1"
        stage_results["stage1"] = _run_stage_command(
            name="stage1",
            idx=2,
            stage_total=stage_total,
            cmd=[
                sys.executable,
                "scripts/run_discovery.py",
                "--config",
                str(pipeline_cfg_path),
                "--run-id",
                stage1_run_id,
                "--candidate-count",
                str(resolved_candidate_count),
                "--seed",
                str(int(args.seed)),
                "--cost-pct",
                str(float(config["costs"]["round_trip_cost_pct"])),
                "--stage-a-months",
                str(int(config["evaluation"]["stage1"]["stage_a_months"])),
                "--stage-b-months",
                str(int(config["evaluation"]["stage1"]["stage_b_months"])),
                "--holdout-months",
                str(int(config["evaluation"]["stage1"]["holdout_months"])),
            ],
            run_dir=run_dir,
            started_at=started_at,
            expected_seconds=900.0 if args.mode == "full" else 420.0,
        )

        if not (Path(args.runs_dir) / stage1_run_id).exists():
            raise FileNotFoundError(f"Expected Stage-1 run folder not found: {stage1_run_id}")

        before_stage2 = _snapshot_suffix_runs(Path(args.runs_dir), "_stage2")
        stage_results["stage2"] = _run_stage_command(
            name="stage2",
            idx=3,
            stage_total=stage_total,
            cmd=[
                sys.executable,
                "scripts/run_stage2.py",
                "--stage1-run-id",
                stage1_run_id,
            ],
            run_dir=run_dir,
            started_at=started_at,
            expected_seconds=240.0,
        )
        stage2_run_id = _detect_new_run_id(Path(args.runs_dir), before_stage2, "_stage2")
        if stage2_run_id is None:
            raise RuntimeError("Failed to detect Stage-2 run id")
        stage_results["stage2"].run_id = stage2_run_id

        before_stage3 = _snapshot_suffix_runs(Path(args.runs_dir), "_stage3_3_selector")
        stage_results["stage3_3"] = _run_stage_command(
            name="stage3_3",
            idx=4,
            stage_total=stage_total,
            cmd=[
                sys.executable,
                "scripts/run_stage3_leverage_selector.py",
                "--config",
                str(pipeline_cfg_path),
                "--stage2-run-id",
                stage2_run_id,
                "--seed",
                str(int(args.seed)),
            ],
            run_dir=run_dir,
            started_at=started_at,
            expected_seconds=420.0,
        )
        stage3_run_id = _detect_new_run_id(Path(args.runs_dir), before_stage3, "_stage3_3_selector")
        if stage3_run_id is None:
            raise RuntimeError("Failed to detect Stage-3.3 run id")
        stage_results["stage3_3"].run_id = stage3_run_id

        before_stage4 = _snapshot_suffix_runs(Path(args.runs_dir), "_stage4")
        stage_results["stage4_spec"] = _run_stage_command(
            name="stage4_spec",
            idx=5,
            stage_total=stage_total,
            cmd=[
                sys.executable,
                "scripts/run_stage4_spec.py",
                "--config",
                str(pipeline_cfg_path),
                "--stage2-run-id",
                stage2_run_id,
                "--stage3-3-run-id",
                stage3_run_id,
            ],
            run_dir=run_dir,
            started_at=started_at,
            expected_seconds=60.0,
        )
        stage4_run_id = _detect_new_run_id(Path(args.runs_dir), before_stage4, "_stage4")
        stage_results["stage4_spec"].run_id = stage4_run_id

        stage4_sim_run_id = None
        if int(args.run_stage4_simulate) == 1:
            before_stage4_sim = _snapshot_suffix_runs(Path(args.runs_dir), "_stage4_sim")
            stage_results["stage4_sim"] = _run_stage_command(
                name="stage4_sim",
                idx=6,
                stage_total=stage_total,
                cmd=[
                    sys.executable,
                    "scripts/run_stage4_simulate.py",
                    "--config",
                    str(pipeline_cfg_path),
                    "--stage2-run-id",
                    stage2_run_id,
                    "--stage3-3-run-id",
                    stage3_run_id,
                    "--seed",
                    str(int(args.seed)),
                ],
                run_dir=run_dir,
                started_at=started_at,
                expected_seconds=180.0,
            )
            stage4_sim_run_id = _detect_new_run_id(Path(args.runs_dir), before_stage4_sim, "_stage4_sim")
            stage_results["stage4_sim"].run_id = stage4_sim_run_id

        selector_summary = _safe_json(Path(args.runs_dir) / stage3_run_id / "selector_summary.json")
        overall = selector_summary.get("overall_choice", {}) if selector_summary else {}
        chosen_method = overall.get("method") or config["evaluation"]["stage4"]["default_method"]
        chosen_leverage = float(overall.get("chosen_leverage") or config["evaluation"]["stage4"]["default_leverage"])

        stage1_summary = _safe_json(Path(args.runs_dir) / stage1_run_id / "summary.json")

        summary_payload = {
            "run_id": run_id,
            "status": "success",
            "error": None,
            "started_at_utc": utc_now_compact(),
            "elapsed_seconds": round(time.time() - started_at, 3),
            "config_path": str(pipeline_cfg_path),
            "config_hash": compute_config_hash(config),
            "data_hash": stage1_summary.get("data_hash"),
            "stage1_run_id": stage1_run_id,
            "stage2_run_id": stage2_run_id,
            "stage3_3_run_id": stage3_run_id,
            "stage4_run_id": stage4_run_id,
            "stage4_sim_run_id": stage4_sim_run_id,
            "chosen_method": chosen_method,
            "chosen_leverage": chosen_leverage,
            "reports": {
                "stage1_report": str(Path(args.runs_dir) / stage1_run_id / "stage1_real_data_report.md"),
                "stage2_report": str(Path(args.runs_dir) / stage2_run_id / "portfolio_report.md"),
                "stage3_3_report": str(Path(args.runs_dir) / stage3_run_id / "selector_report.md"),
                "trading_spec": str(PROJECT_ROOT / "docs" / "trading_spec.md"),
                "paper_checklist": str(PROJECT_ROOT / "docs" / "paper_trading_checklist.md"),
                "stage4_sim_metrics": (
                    str(Path(args.runs_dir) / stage4_sim_run_id / "execution_metrics.json")
                    if stage4_sim_run_id
                    else None
                ),
            },
        }
        _write_json_atomic(run_dir / "pipeline_summary.json", summary_payload)
        build_ui_bundle_from_pipeline(run_dir)

        _update_progress(
            path=run_dir / "progress.json",
            stage="done",
            stage_idx=stage_total,
            stage_total=stage_total,
            stage_progress_pct=100.0,
            elapsed_seconds=time.time() - started_at,
            eta_seconds=0.0,
            counters={"resolved_candidate_count": int(resolved_candidate_count)},
            status="done",
            last_log_line="Pipeline completed successfully.",
        )
        print(f"pipeline_run_id: {run_id}")
        print("status: success")

    except PipelineCancelled as exc:
        _write_json_atomic(
            run_dir / "pipeline_summary.json",
            {
                "run_id": run_id,
                "status": "cancelled",
                "error": str(exc),
                "elapsed_seconds": round(time.time() - started_at, 3),
                "stage1_run_id": stage1_run_id,
                "stage2_run_id": stage2_run_id,
                "stage3_3_run_id": stage3_run_id,
                "stage4_run_id": stage4_run_id,
                "stage4_sim_run_id": stage4_sim_run_id,
            },
        )
        try:
            build_ui_bundle_from_pipeline(run_dir)
        except Exception:
            pass
        _update_progress(
            path=run_dir / "progress.json",
            stage="cancelled",
            stage_idx=0,
            stage_total=stage_total,
            stage_progress_pct=0.0,
            elapsed_seconds=time.time() - started_at,
            eta_seconds=0.0,
            counters={},
            status="cancelled",
            last_log_line=str(exc),
        )
        print(f"pipeline_run_id: {run_id}")
        print("status: cancelled")
        raise SystemExit(2) from exc
    except Exception as exc:
        _write_json_atomic(
            run_dir / "pipeline_summary.json",
            {
                "run_id": run_id,
                "status": "failed",
                "error": str(exc),
                "elapsed_seconds": round(time.time() - started_at, 3),
                "stage1_run_id": stage1_run_id,
                "stage2_run_id": stage2_run_id,
                "stage3_3_run_id": stage3_run_id,
                "stage4_run_id": stage4_run_id,
                "stage4_sim_run_id": stage4_sim_run_id,
                "config_hash": compute_config_hash(config) if config else "",
                "seed": int(args.seed),
            },
        )
        try:
            build_ui_bundle_from_pipeline(run_dir)
        except Exception:
            pass
        _update_progress(
            path=run_dir / "progress.json",
            stage="failed",
            stage_idx=0,
            stage_total=stage_total,
            stage_progress_pct=0.0,
            elapsed_seconds=time.time() - started_at,
            eta_seconds=0.0,
            counters={},
            status="failed",
            last_log_line=str(exc),
        )
        print(f"pipeline_run_id: {run_id}")
        print(f"status: failed: {exc}")
        raise
    finally:
        release_lock(run_id=run_id, runs_dir=Path(args.runs_dir))


def _prepare_config(config: dict[str, Any], args: argparse.Namespace, symbols: list[str]) -> tuple[dict[str, Any], int]:
    cfg = deepcopy(config)

    ui_stage5 = cfg.get("ui", {}).get("stage5", {})
    presets = ui_stage5.get("presets", {}) if isinstance(ui_stage5, dict) else {}
    quick_cfg = presets.get("quick", {}) if isinstance(presets, dict) else {}
    full_cfg = presets.get("full", {}) if isinstance(presets, dict) else {}

    base_candidate_count = int(cfg["evaluation"]["stage1"]["candidate_count"])
    quick_candidate_default = int(quick_cfg.get("candidate_count", min(1000, base_candidate_count)))
    full_candidate_default = int(full_cfg.get("candidate_count", base_candidate_count))

    if args.candidate_count is not None:
        resolved_candidate_count = int(args.candidate_count)
    elif str(args.mode) == "quick":
        resolved_candidate_count = quick_candidate_default
    else:
        resolved_candidate_count = full_candidate_default

    resolved_candidate_count = max(1, int(resolved_candidate_count))

    cfg["universe"]["symbols"] = symbols
    cfg["universe"]["timeframe"] = str(args.timeframe)
    cfg["search"]["seed"] = int(args.seed)
    cfg["execution"]["mode"] = str(args.execution_mode)

    if args.fees_round_trip_pct is not None:
        cfg["costs"]["round_trip_cost_pct"] = float(args.fees_round_trip_pct)

    stage1 = cfg["evaluation"]["stage1"]
    stage1["candidate_count"] = int(resolved_candidate_count)
    stage1["holdout_months"] = int(args.window_months)
    stage1["stage_a_months"] = int(max(3, min(int(args.window_months), 9)))
    stage1["stage_b_months"] = int(max(int(stage1["stage_a_months"]), int(args.window_months)))

    stage1["top_k"] = int(min(int(stage1["top_k"]), int(stage1["candidate_count"])))
    stage1["top_m"] = int(min(int(stage1["top_m"]), int(stage1["top_k"])))

    cfg["evaluation"]["stage06"]["window_months"] = int(args.window_months)

    if str(args.mode) == "quick":
        selector = cfg["portfolio"]["leverage_selector"]
        selector["n_paths"] = int(min(int(selector["n_paths"]), 5000))
        selector["leverage_levels"] = [1, 2, 3, 5, 10]

    cfg["evaluation"]["stage4"]["default_method"] = "equal"
    cfg["evaluation"]["stage4"]["default_leverage"] = 1.0

    return cfg, resolved_candidate_count


def _run_data_validate(
    symbols: list[str],
    timeframe: str,
    run_dir: Path,
    started_at: float,
    stage_idx: int,
    stage_total: int,
) -> None:
    _update_progress(
        path=run_dir / "progress.json",
        stage="data_validate",
        stage_idx=stage_idx,
        stage_total=stage_total,
        stage_progress_pct=10.0,
        elapsed_seconds=time.time() - started_at,
        eta_seconds=0.0,
        counters={},
        status="running",
        last_log_line="Validating local parquet data availability.",
    )

    missing: list[str] = []
    found = 0
    for symbol in symbols:
        path = parquet_path(symbol=symbol, timeframe=timeframe)
        if path.exists():
            found += 1
        else:
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing required local parquet files. Offline pipeline cannot fetch data: " + ", ".join(missing)
        )

    _update_progress(
        path=run_dir / "progress.json",
        stage="data_validate",
        stage_idx=stage_idx,
        stage_total=stage_total,
        stage_progress_pct=100.0,
        elapsed_seconds=time.time() - started_at,
        eta_seconds=0.0,
        counters={"data_files_found": found},
        status="running",
        last_log_line="Data validation passed.",
    )
    print("[data_validate] local parquet files verified")


def _run_stage_command(
    name: str,
    idx: int,
    stage_total: int,
    cmd: list[str],
    run_dir: Path,
    started_at: float,
    expected_seconds: float,
) -> StageResult:
    log_path = run_dir / f"{name}.log"
    cancel_flag = run_dir / "cancel.flag"

    print(f"[{name}] starting: {' '.join(cmd)}")
    stage_started = time.time()
    last_emitted_line = ""
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
        )

        while True:
            if cancel_flag.exists():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise PipelineCancelled(f"Pipeline cancelled during {name}.")

            ret = process.poll()
            elapsed = time.time() - stage_started
            pct = 100.0 if ret is not None else min(95.0, (elapsed / max(expected_seconds, 1.0)) * 100.0)
            eta = max(0.0, expected_seconds - elapsed) if ret is None else 0.0
            last_line = _tail_last_line(log_path) or f"{name} running"

            _update_progress(
                path=run_dir / "progress.json",
                stage=name,
                stage_idx=idx,
                stage_total=stage_total,
                stage_progress_pct=pct,
                elapsed_seconds=time.time() - started_at,
                eta_seconds=eta,
                counters={"stage_elapsed_seconds": round(elapsed, 2)},
                status="running",
                last_log_line=last_line,
            )

            if last_line != last_emitted_line:
                print(f"[{name}] {last_line}")
                last_emitted_line = last_line

            if ret is not None:
                if ret != 0:
                    raise RuntimeError(f"Stage `{name}` failed with exit code {ret}. See {log_path}")
                break
            time.sleep(2)

    _update_progress(
        path=run_dir / "progress.json",
        stage=name,
        stage_idx=idx,
        stage_total=stage_total,
        stage_progress_pct=100.0,
        elapsed_seconds=time.time() - started_at,
        eta_seconds=0.0,
        counters={},
        status="running",
        last_log_line=f"{name} completed",
    )
    print(f"[{name}] completed")
    return StageResult(name=name, run_id=None, log_path=log_path)


def _snapshot_suffix_runs(runs_dir: Path, suffix: str) -> set[str]:
    return {path.name for path in runs_dir.glob(f"*{suffix}") if path.is_dir()}


def _detect_new_run_id(runs_dir: Path, before: set[str], suffix: str) -> str | None:
    after = _snapshot_suffix_runs(runs_dir, suffix)
    new = sorted(after - before)
    if len(new) == 1:
        return new[0]
    if len(new) > 1:
        latest = max((runs_dir / item for item in new), key=lambda path: path.stat().st_mtime)
        return latest.name

    candidates = [path for path in runs_dir.glob(f"*{suffix}") if path.is_dir()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest.name


def _update_progress(
    path: Path,
    stage: str,
    stage_idx: int,
    stage_total: int,
    stage_progress_pct: float,
    elapsed_seconds: float,
    eta_seconds: float,
    counters: dict[str, Any],
    status: str,
    last_log_line: str,
) -> None:
    payload = {
        "stage": str(stage),
        "stage_idx": int(stage_idx),
        "stage_total": int(stage_total),
        "stage_progress_pct": float(stage_progress_pct),
        "elapsed_seconds": float(round(elapsed_seconds, 3)),
        "eta_seconds": float(round(eta_seconds, 3)),
        "counters": counters or {},
        "status": str(status),
        "last_log_line": str(last_log_line),
    }
    _write_json_atomic(path, payload)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail_last_line(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    if not lines:
        return ""
    return lines[-1][-500:]


if __name__ == "__main__":
    main()
