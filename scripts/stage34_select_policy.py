"""Stage-34.5 policy selection and replay runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage34.policy import PolicyConfig, replay_policy, select_best_policy
from buffmini.stage34.train import load_models
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Stage-34 policy and replay")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-rows-path", type=str, default="")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--dataset-meta-path", type=str, default="")
    parser.add_argument("--models-dir", type=str, default="")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def _latest(path_glob: str, runs_dir: Path) -> Path:
    items = sorted(Path(runs_dir).glob(path_glob))
    if not items:
        raise FileNotFoundError(f"No artifact found for pattern: {path_glob}")
    return items[-1]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage_cfg = ((cfg.get("evaluation", {}) or {}).get("stage34", {})) or {}
    eval_cfg = (stage_cfg.get("evaluation", {})) or {}

    eval_rows_path = Path(str(args.eval_rows_path).strip()) if str(args.eval_rows_path).strip() else _latest("*_stage34_eval/stage34/eval/eval_rows.csv", args.runs_dir)
    dataset_path = Path(str(args.dataset_path).strip()) if str(args.dataset_path).strip() else _latest("*_stage34_ds/stage34/dataset/dataset.parquet", args.runs_dir)
    dataset_meta_path = Path(str(args.dataset_meta_path).strip()) if str(args.dataset_meta_path).strip() else dataset_path.with_name("dataset_meta.json")
    models_dir = Path(str(args.models_dir).strip()) if str(args.models_dir).strip() else _latest("*_stage34_train/stage34/models", args.runs_dir)

    eval_rows = pd.read_csv(eval_rows_path)
    dataset = pd.read_parquet(dataset_path)
    ds_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    models = load_models(models_dir=models_dir)

    cfg_policy = PolicyConfig(
        threshold=float(eval_cfg.get("threshold", 0.55)),
        risk_cap=0.20,
        equity=10_000.0,
    )
    policy = select_best_policy(eval_rows, cfg=cfg_policy, seed=int(args.seed))
    model_name = str(policy.get("model_name", ""))
    if model_name and model_name in models:
        selected_model = models[model_name]
    else:
        selected_model = models[sorted(models.keys())[0]]
        policy["model_name"] = sorted(models.keys())[0]

    split_idx = int(max(10, round(dataset.shape[0] * 0.8)))
    heldout = dataset.iloc[split_idx:, :].reset_index(drop=True)
    replay_research = replay_policy(heldout, model=selected_model, policy=policy, mode="research", cfg=cfg_policy)
    replay_live = replay_policy(heldout, model=selected_model, policy=policy, mode="live", cfg=cfg_policy)

    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': int(args.seed), 'cfg': compute_config_hash(cfg), 'data': ds_meta.get('data_hash', ''), 'policy': policy.get('policy_id', ''), 'model': policy.get('model_name', '')}, length=12)}"
        "_stage34_policy"
    )
    out_dir = Path(args.runs_dir) / run_id / "stage34" / "policy"
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = out_dir / "policy_snapshot.json"
    replay_path = out_dir / "policy_replay_summary.json"
    policy_path.write_text(json.dumps(policy, indent=2, allow_nan=False), encoding="utf-8")
    replay_payload = {
        "run_id": run_id,
        "policy": policy,
        "research": replay_research,
        "live": replay_live,
        "accepted_rejected_breakdown": replay_live.get("accepted_rejected_breakdown", {}),
        "top_reject_reasons": replay_live.get("top_reject_reasons", []),
    }
    replay_path.write_text(json.dumps(replay_payload, indent=2, allow_nan=False), encoding="utf-8")

    summary = {
        "stage": "34.5",
        "run_id": run_id,
        "policy_id": policy.get("policy_id", ""),
        "model_name": policy.get("model_name", ""),
        "policy_status": policy.get("status", ""),
        "research_trade_count": replay_research.get("trade_count", 0),
        "live_trade_count": replay_live.get("trade_count", 0),
        "accepted_rejected_breakdown": replay_live.get("accepted_rejected_breakdown", {}),
        "top_reject_reasons": replay_live.get("top_reject_reasons", []),
        "config_hash": compute_config_hash(cfg),
        "data_hash": ds_meta.get("data_hash", ""),
        "resolved_end_ts": ds_meta.get("resolved_end_ts"),
        **snapshot_metadata_from_config(cfg),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"policy_snapshot: {policy_path}")
    print(f"policy_replay_summary: {replay_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
