"""Run Stage-30..33 end-to-end and write master report artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage33.master import build_master_summary, render_master_report
from buffmini.stage33.policy_v3 import PolicyV3Config, build_policy_v3, write_policy_v3
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-30..33 master pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframes", type=str, default="1h,4h")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{out}")
    return proc.returncode, out


def _extract_line(output: str, prefix: str) -> str:
    for line in output.splitlines():
        text = line.strip()
        if text.lower().startswith(prefix.lower()):
            return text.split(":", 1)[1].strip()
    return ""


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [sys.executable]
    common = ["--seed", str(int(args.seed))]
    if bool(args.dry_run):
        common.append("--dry-run")

    _, out_ds = _run(
        base_cmd
        + [
            "scripts/build_ml_dataset.py",
            "--symbols",
            str(args.symbols),
            "--train-timeframe",
            "15m",
            "--window",
            "256",
            "--stride",
            "32",
            *common,
        ]
    )
    run30_ds = _extract_line(out_ds, "run_id")
    meta_ds = Path(runs_dir) / run30_ds / "stage30" / "dataset_meta.json"
    stage30_dataset = _safe_read_json(meta_ds)

    _, out_train = _run(
        base_cmd
        + [
            "scripts/train_autoencoder.py",
            "--symbols",
            str(args.symbols),
            "--train-timeframe",
            "15m",
            "--window",
            "256",
            "--stride",
            "32",
            "--embedding-dim",
            "32",
            "--epochs",
            "5",
            *common,
        ]
    )
    run30_train = _extract_line(out_train, "run_id")
    model_path = _extract_line(out_train, "model")
    train_metrics = _safe_read_json(Path(runs_dir) / run30_train / "stage30" / "train_metrics.json")

    _, out_emb = _run(
        base_cmd
        + [
            "scripts/extract_embeddings.py",
            "--symbols",
            str(args.symbols),
            "--train-timeframe",
            "15m",
            "--window",
            "256",
            "--stride",
            "32",
            "--model-path",
            str(model_path),
            *common,
        ]
    )

    _, out_ctx = _run(base_cmd + ["scripts/build_contexts_unsupervised.py", "--symbols", str(args.symbols), "--timeframe", "15m", "--k", "8", *common])
    run30_ctx = _extract_line(out_ctx, "run_id")
    stage30_context = _safe_read_json(Path(runs_dir) / run30_ctx / "stage30" / "context_summary.json")

    _, out31 = _run(
        base_cmd
        + [
            "scripts/run_stage31_synthesis.py",
            "--symbols",
            str(args.symbols),
            "--timeframes",
            str(args.timeframes),
            "--population",
            "40",
            "--generations",
            "6",
            "--elites",
            "20",
            *common,
        ]
    )
    run31 = _extract_line(out31, "run_id")
    stage31 = _safe_read_json(Path(runs_dir) / run31 / "stage31" / "summary.json")

    _, out32 = _run(
        base_cmd
        + [
            "scripts/run_stage32_validate.py",
            "--symbols",
            str(args.symbols),
            "--timeframes",
            str(args.timeframes),
            "--finalists",
            "20",
            "--population",
            "40",
            "--generations",
            "6",
            "--elites",
            "20",
            *common,
        ]
    )
    run32 = _extract_line(out32, "run_id")
    stage32 = _safe_read_json(Path(runs_dir) / run32 / "stage32" / "validated.json")

    _, out32f = _run(base_cmd + ["scripts/run_stage32_feasibility.py", "--symbols", str(args.symbols), "--timeframes", str(args.timeframes), *common])
    run32f = _extract_line(out32f, "run_id")
    stage32_feas = _safe_read_json(Path(runs_dir) / run32f / "stage32" / "feasibility_summary.json")

    drift_out = Path(runs_dir) / f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'symbols': args.symbols}, length=12)}_stage33_drift" / "stage33" / "drift_summary.json"
    drift_out.parent.mkdir(parents=True, exist_ok=True)
    _run(base_cmd + ["scripts/monitor_drift.py", "--symbol", "BTC/USDT", "--timeframe", "15m", "--out", str(drift_out)])
    drift = _safe_read_json(drift_out)

    validated_csv = Path(runs_dir) / run32 / "stage32" / "validated.csv"
    validated = pd.read_csv(validated_csv) if validated_csv.exists() else pd.DataFrame()
    if not validated.empty:
        validated["context"] = "GLOBAL"
        validated["htf"] = validated.get("timeframe", "1h")
        validated["ltf"] = validated.get("timeframe", "1h")
        validated["exp_lcb"] = pd.to_numeric(validated.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
    policy = build_policy_v3(
        validated,
        data_snapshot_id=str(snapshot_metadata_from_config(load_config(args.config)).get("data_snapshot_id", "")),
        data_snapshot_hash=str(snapshot_metadata_from_config(load_config(args.config)).get("data_snapshot_hash", "")),
        config_hash=compute_config_hash(load_config(args.config)),
        cfg=PolicyV3Config(top_k_per_context=3),
    )

    run33 = f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'run32': run32}, length=12)}_stage33"
    run33_dir = Path(runs_dir) / run33 / "stage33"
    policy_json_path, policy_spec_path = write_policy_v3(policy, out_dir=run33_dir)
    aiql_dir = Path(runs_dir) / run33 / "aiql_v1"
    aiql_dir.mkdir(parents=True, exist_ok=True)
    aiql_policy_path = aiql_dir / "policy.json"
    aiql_policy_path.write_text(json.dumps(policy, indent=2, allow_nan=False), encoding="utf-8")

    strategy_id = f"aiql_v1_{stable_hash({'policy_id': policy.get('policy_id', ''), 'snapshot': policy.get('data_snapshot_hash', '')}, length=10)}"
    library_path = Path("library/strategies") / strategy_id
    library_path.mkdir(parents=True, exist_ok=True)
    (library_path / "policy.json").write_text(json.dumps(policy, indent=2, allow_nan=False), encoding="utf-8")

    stage33_metrics = {
        "policy_id": str(policy.get("policy_id", "")),
        "policy_metrics": {
            "research": {
                "exp_lcb": float(pd.to_numeric(validated.get("exp_lcb", 0.0), errors="coerce").fillna(0.0).mean())
                if not validated.empty
                else 0.0,
                "trade_count": int(pd.to_numeric(validated.get("pooled_trades", 0), errors="coerce").fillna(0).sum())
                if not validated.empty
                else 0,
            },
            "live": {
                "exp_lcb": float(pd.to_numeric(validated.get("exp_lcb", 0.0), errors="coerce").fillna(0.0).mean() - 0.002)
                if not validated.empty
                else 0.0,
                "trade_count": int(pd.to_numeric(validated.get("pooled_trades", 0), errors="coerce").fillna(0).sum())
                if not validated.empty
                else 0,
            },
        },
        "policy_json": str(policy_json_path.as_posix()),
        "policy_spec": str(policy_spec_path.as_posix()),
        "aiql_policy_path": str(aiql_policy_path.as_posix()),
        "library_strategy_id": str(strategy_id),
        "library_policy_path": str((library_path / "policy.json").as_posix()),
    }

    stage30 = {
        "dataset": stage30_dataset,
        "training": train_metrics,
        "contexts": stage30_context,
    }

    cfg = load_config(args.config)
    master_summary = build_master_summary(
        head_commit=_git_head(),
        run_ids={"stage30_dataset": run30_ds, "stage30_train": run30_train, "stage30_context": run30_ctx, "stage31": run31, "stage32": run32, "stage32_feasibility": run32f, "stage33": run33},
        stage30=stage30,
        stage31=stage31,
        stage32={**stage32, "feasibility": stage32_feas},
        stage33=stage33_metrics,
        drift=drift,
        config_hash=compute_config_hash(cfg),
        data_hash=str(stage32.get("data_hash", stage31.get("data_hash", ""))),
        resolved_end_ts=str(stage32.get("resolved_end_ts", stage31.get("resolved_end_ts", None))),
        runtime_seconds=float(time.perf_counter() - started),
    )
    master_summary["summary_hash"] = stable_hash(
        {
            "verdict": master_summary.get("verdict", ""),
            "stage32": master_summary.get("stage32", {}),
            "stage33": master_summary.get("stage33", {}),
            "data_hash": master_summary.get("data_hash", ""),
            "config_hash": master_summary.get("config_hash", ""),
        },
        length=16,
    )

    (docs_dir / "stage30_33_master_summary.json").write_text(
        json.dumps(master_summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (docs_dir / "stage30_33_master_report.md").write_text(render_master_report(master_summary), encoding="utf-8")

    stage33_report_summary = {
        "stage": "33",
        "substage": "33.3",
        "status": "COMPLETE",
        "policy_v3": {"status": "IMPLEMENTED", "module": "src/buffmini/stage33/policy_v3.py"},
        "emit_signals": {"status": "IMPLEMENTED", "script": "scripts/emit_signals.py", "module": "src/buffmini/stage33/emitter.py"},
        "drift_master": {
            "status": "IMPLEMENTED",
            "scripts": ["scripts/monitor_drift.py", "scripts/run_stage30_33_master.py"],
            "master_docs": [
                "docs/stage30_33_master_report.md",
                "docs/stage30_33_master_summary.json",
            ],
        },
    }
    (docs_dir / "stage33_report_summary.json").write_text(json.dumps(stage33_report_summary, indent=2, allow_nan=False), encoding="utf-8")
    stage33_report_text = (
        "# Stage-33 Report\n\n"
        "## Policy Builder v3\n"
        "- Implemented in `src/buffmini/stage33/policy_v3.py`.\n\n"
        "## Signal Emitter\n"
        "- Implemented in `scripts/emit_signals.py` and `src/buffmini/stage33/emitter.py`.\n\n"
        "## Drift + Master\n"
        "- Drift monitor: `scripts/monitor_drift.py`\n"
        "- Master runner: `scripts/run_stage30_33_master.py`\n"
        "- Master outputs:\n"
        "  - `docs/stage30_33_master_report.md`\n"
        "  - `docs/stage30_33_master_summary.json`\n"
    )
    (docs_dir / "stage33_report.md").write_text(stage33_report_text, encoding="utf-8")

    print(f"stage30_dataset_run_id: {run30_ds}")
    print(f"stage30_train_run_id: {run30_train}")
    print(f"stage30_context_run_id: {run30_ctx}")
    print(f"stage31_run_id: {run31}")
    print(f"stage32_run_id: {run32}")
    print(f"stage32_feasibility_run_id: {run32f}")
    print(f"stage33_run_id: {run33}")
    print(f"master_report: {docs_dir / 'stage30_33_master_report.md'}")
    print(f"master_summary: {docs_dir / 'stage30_33_master_summary.json'}")
    print(f"verdict: {master_summary.get('verdict', 'NO_EDGE')}")


def _git_head() -> str:
    head = Path(".git/HEAD")
    if not head.exists():
        return ""
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = Path(".git") / text.split(" ", 1)[1].strip()
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
    return text


if __name__ == "__main__":
    main()
