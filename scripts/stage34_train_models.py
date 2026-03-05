"""Stage-34.3 deterministic model training runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage34.train import TrainConfig, save_models, train_stage34_models
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-34 models")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--dataset-meta-path", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--calibration", type=str, default="")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    text = str(value).strip()
    return [item.strip() for item in text.split(",") if item.strip()] if text else []


def _find_latest_dataset(runs_dir: Path) -> tuple[Path, Path]:
    candidates = sorted(Path(runs_dir).glob("*_stage34_ds/stage34/dataset/dataset.parquet"))
    if not candidates:
        raise FileNotFoundError("No Stage-34 dataset found. Run scripts/stage34_build_dataset.py first.")
    dataset_path = candidates[-1]
    meta_path = dataset_path.with_name("dataset_meta.json")
    return dataset_path, meta_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage_cfg = ((cfg.get("evaluation", {}) or {}).get("stage34", {})) or {}
    train_cfg = (stage_cfg.get("training", {})) or {}
    models = tuple(_csv(args.models) or [str(v) for v in train_cfg.get("models", ["logreg", "hgbt", "rf"])])
    calibration = str(args.calibration or train_cfg.get("calibration", "platt"))

    if str(args.dataset_path).strip():
        dataset_path = Path(str(args.dataset_path).strip())
        meta_path = Path(str(args.dataset_meta_path).strip()) if str(args.dataset_meta_path).strip() else dataset_path.with_name("dataset_meta.json")
    else:
        dataset_path, meta_path = _find_latest_dataset(Path(args.runs_dir))

    dataset = pd.read_parquet(dataset_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feat_cols = [str(v) for v in meta.get("feature_columns", []) if str(v) in dataset.columns]
    if not feat_cols:
        raise ValueError("No feature columns available in dataset meta")

    models_bundle, summary = train_stage34_models(
        dataset,
        feature_columns=feat_cols,
        cfg=TrainConfig(
            seed=int(args.seed),
            models=models,
            calibration=calibration,
        ),
    )

    config_hash = compute_config_hash(cfg)
    data_hash = str(meta.get("data_hash", ""))
    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': int(args.seed), 'cfg': config_hash, 'data': data_hash, 'dataset_hash': meta.get('dataset_hash', ''), 'models': list(models), 'cal': calibration}, length=12)}"
        "_stage34_train"
    )
    out_dir = Path(args.runs_dir) / run_id / "stage34" / "models"
    model_paths = save_models(models_bundle, out_dir=out_dir)

    payload: dict[str, Any] = {
        "stage": "34.3",
        "run_id": run_id,
        "seed": int(args.seed),
        "models": list(models),
        "calibration": calibration,
        "dataset_path": str(dataset_path.as_posix()),
        "dataset_meta_path": str(meta_path.as_posix()),
        "model_paths": {k: str(v.as_posix()) for k, v in model_paths.items()},
        "feature_columns": feat_cols,
        "config_hash": config_hash,
        "data_hash": data_hash,
        "resolved_end_ts": meta.get("resolved_end_ts"),
        **summary,
        **snapshot_metadata_from_config(cfg),
    }
    summary_path = Path(args.runs_dir) / run_id / "stage34" / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    card = _render_model_card(payload)
    card_path = Path("docs/stage34_ml_model_card.md")
    card_path.write_text(card, encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"train_summary: {summary_path}")
    print(f"model_card: {card_path}")


def _render_model_card(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage-34 ML Model Card",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- models: `{summary.get('models', [])}`",
        f"- calibration: `{summary.get('calibration', '')}`",
        f"- rows_total: `{int(summary.get('rows_total', 0))}`",
        f"- split train/val/test: `{summary.get('splits', {})}`",
        "",
        "## Hyperparameters",
        "- logreg: gradient descent with L2 regularization",
        "- hgbt: deterministic boosted stump ensemble",
        "- rf: deterministic bagged stump ensemble",
        "",
        "## Metrics Summary",
    ]
    for model in summary.get("models", []):
        lines.append(
            "- `{name}`: val_logloss={vll:.6f}, test_logloss={tll:.6f}, val_brier={vb:.6f}, test_brier={tb:.6f}".format(
                name=str(model.get("model_name", "")),
                vll=float(model.get("val_logloss", 0.0)),
                tll=float(model.get("test_logloss", 0.0)),
                vb=float(model.get("val_brier", 0.0)),
                tb=float(model.get("test_brier", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Train/Validation Time Safety",
            "- Time-ordered split only (no shuffle).",
            "- Calibration fitted on validation split only.",
            "",
            "## Limitations",
            "- CPU-first lightweight models only.",
            "- Probabilities are calibrated but not guaranteed well-calibrated in sparse contexts.",
            "",
            "## Runtime Budget",
            "- Designed for laptop execution with bounded estimators and deterministic seeds.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    main()
