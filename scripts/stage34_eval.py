"""Stage-34.4 strict evaluation runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage34.eval import EvalConfig, evaluate_models_strict
from buffmini.stage34.train import load_models
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-34 strict evaluation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--dataset-meta-path", type=str, default="")
    parser.add_argument("--models-dir", type=str, default="")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--window-months", type=str, default="")
    parser.add_argument("--step-months", type=int, default=-1)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def _csv_int(value: str) -> list[int]:
    text = str(value).strip()
    if not text:
        return []
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _find_latest_dataset(runs_dir: Path) -> tuple[Path, Path]:
    items = sorted(Path(runs_dir).glob("*_stage34_ds/stage34/dataset/dataset.parquet"))
    if not items:
        raise FileNotFoundError("No stage34 dataset found")
    path = items[-1]
    return path, path.with_name("dataset_meta.json")


def _find_latest_models(runs_dir: Path) -> Path:
    items = sorted(Path(runs_dir).glob("*_stage34_train/stage34/models"))
    if not items:
        raise FileNotFoundError("No stage34 models found")
    return items[-1]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage_cfg = ((cfg.get("evaluation", {}) or {}).get("stage34", {})) or {}
    eval_cfg = (stage_cfg.get("evaluation", {})) or {}
    threshold = float(args.threshold if args.threshold > 0 else eval_cfg.get("threshold", 0.55))
    window_months = tuple(_csv_int(args.window_months) or [int(v) for v in eval_cfg.get("window_months", [3, 6])])
    step_months = int(args.step_months if args.step_months > 0 else eval_cfg.get("step_months", 1))
    min_usable = int(eval_cfg.get("min_usable_windows", 3))
    mc_min_trades = int(eval_cfg.get("mc_min_trades", 30))

    dataset_path, meta_path = (
        (Path(str(args.dataset_path).strip()), Path(str(args.dataset_meta_path).strip()) if str(args.dataset_meta_path).strip() else Path(str(args.dataset_path).strip()).with_name("dataset_meta.json"))
        if str(args.dataset_path).strip()
        else _find_latest_dataset(Path(args.runs_dir))
    )
    models_dir = Path(str(args.models_dir).strip()) if str(args.models_dir).strip() else _find_latest_models(Path(args.runs_dir))

    dataset = pd.read_parquet(dataset_path)
    ds_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    models = load_models(models_dir=models_dir)
    rows, summary = evaluate_models_strict(
        dataset,
        models=models,
        cfg=EvalConfig(
            threshold=float(threshold),
            window_months=window_months,
            step_months=int(step_months),
            min_usable_windows=int(min_usable),
            mc_min_trades=int(mc_min_trades),
            seed=int(args.seed),
        ),
    )

    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': int(args.seed), 'cfg': compute_config_hash(cfg), 'data': ds_meta.get('data_hash', ''), 'models': sorted(list(models.keys())), 'thr': threshold, 'windows': list(window_months), 'step': int(step_months)}, length=12)}"
        "_stage34_eval"
    )
    out_dir = Path(args.runs_dir) / run_id / "stage34" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "eval_rows.csv"
    rows.to_csv(rows_path, index=False)

    payload: dict[str, Any] = {
        "stage": "34.4",
        "run_id": run_id,
        "seed": int(args.seed),
        "dataset_path": str(dataset_path.as_posix()),
        "models_dir": str(models_dir.as_posix()),
        "threshold": float(threshold),
        "window_months": [int(v) for v in window_months],
        "step_months": int(step_months),
        "final_verdict": str(summary.get("final_verdict", "NO_EDGE")),
        "wf_executed_pct": float(summary.get("wf_executed_pct", 0.0)),
        "mc_trigger_pct": float(summary.get("mc_trigger_pct", 0.0)),
        "failure_mode_counts": summary.get("failure_mode_counts", {}),
        "config_hash": compute_config_hash(cfg),
        "data_hash": ds_meta.get("data_hash", ""),
        "resolved_end_ts": ds_meta.get("resolved_end_ts"),
        **snapshot_metadata_from_config(cfg),
    }
    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    failure_doc = _render_failure_modes(rows, payload)
    failure_path = Path("docs/stage34_failure_modes.md")
    failure_path.write_text(failure_doc, encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"eval_rows: {rows_path}")
    print(f"eval_summary: {summary_path}")
    print(f"failure_doc: {failure_path}")


def _render_failure_modes(rows: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# Stage-34 Failure Modes",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- final_verdict: `{summary.get('final_verdict', '')}`",
        f"- wf_executed_pct: `{float(summary.get('wf_executed_pct', 0.0)):.2f}`",
        f"- mc_trigger_pct: `{float(summary.get('mc_trigger_pct', 0.0)):.2f}`",
        "",
        "## Failure Counts",
    ]
    counts = dict(summary.get("failure_mode_counts", {}))
    if counts:
        for key in sorted(counts.keys()):
            lines.append(f"- `{key}`: `{int(counts[key])}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Top Rows"])
    if rows.empty:
        lines.append("- no evaluation rows")
    else:
        top = rows.sort_values("exp_lcb", ascending=False).head(10)
        lines.append("| model | cost | window_m | trades | exp_lcb | pf_adj | maxdd_p95 | wf | mc | failure |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")
        for rec in top.to_dict(orient="records"):
            lines.append(
                "| {model_name} | {cost_mode} | {window_months} | {trade_count} | {exp_lcb:.6f} | {pf_adj:.6f} | {maxdd_p95:.6f} | {wf_executed} | {mc_triggered} | {failure_mode} |".format(
                    model_name=str(rec.get("model_name", "")),
                    cost_mode=str(rec.get("cost_mode", "")),
                    window_months=int(rec.get("window_months", 0)),
                    trade_count=int(rec.get("trade_count", 0)),
                    exp_lcb=float(rec.get("exp_lcb", 0.0)),
                    pf_adj=float(rec.get("pf_adj", 0.0)),
                    maxdd_p95=float(rec.get("maxdd_p95", 0.0)),
                    wf_executed=bool(rec.get("wf_executed", False)),
                    mc_triggered=bool(rec.get("mc_triggered", False)),
                    failure_mode=str(rec.get("failure_mode", "")),
                )
            )
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    main()
