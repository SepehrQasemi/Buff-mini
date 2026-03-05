"""Stage-34.7/34.8 deterministic generation loop runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage34.evolution import EvolutionConfig, run_evolution
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-34 evolution generations")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--budget", type=str, default="")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--dataset-meta-path", type=str, default="")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def _latest_dataset(runs_dir: Path) -> tuple[Path, Path]:
    items = sorted(Path(runs_dir).glob("*_stage34_ds/stage34/dataset/dataset.parquet"))
    if not items:
        raise FileNotFoundError("No stage34 dataset found")
    ds = items[-1]
    return ds, ds.with_name("dataset_meta.json")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage_cfg = ((cfg.get("evaluation", {}) or {}).get("stage34", {})) or {}
    evo_cfg = (stage_cfg.get("evolution", {})) or {}
    budget = str(args.budget or evo_cfg.get("budget", "small")).strip().lower()
    max_models = int(evo_cfg.get("max_models_per_generation", 12))
    if budget == "small":
        max_models = min(max_models, 8)
    elif budget == "medium":
        max_models = min(max_models, 12)

    dataset_path, meta_path = (
        (Path(str(args.dataset_path).strip()), Path(str(args.dataset_meta_path).strip()) if str(args.dataset_meta_path).strip() else Path(str(args.dataset_path).strip()).with_name("dataset_meta.json"))
        if str(args.dataset_path).strip()
        else _latest_dataset(Path(args.runs_dir))
    )
    dataset = pd.read_parquet(dataset_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if {"symbol", "timeframe"}.issubset(set(dataset.columns)):
        preferred = dataset.loc[
            (dataset["symbol"].astype(str) == "BTC/USDT") & (dataset["timeframe"].astype(str) == "1h")
        ].copy()
        if preferred.empty:
            preferred = dataset.loc[dataset["timeframe"].astype(str) == "1h"].copy()
        if preferred.empty:
            preferred = dataset.copy()
        dataset = preferred.reset_index(drop=True)
    row_cap = 20_000 if budget == "small" else 60_000
    if dataset.shape[0] > row_cap:
        dataset = dataset.tail(int(row_cap)).reset_index(drop=True)
    feat_pool = [c for c in dataset.columns if c not in {"timestamp", "symbol", "timeframe", "label_primary", "label_auxiliary", "open", "high", "low", "close", "volume"}]

    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': int(args.seed), 'cfg': compute_config_hash(cfg), 'data': meta.get('data_hash', ''), 'dataset_hash': meta.get('dataset_hash', ''), 'gens': int(args.generations), 'budget': budget}, length=12)}"
        "_stage34_gen"
    )
    out_dir = Path(args.runs_dir) / run_id / "stage34" / "generations"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = out_dir / "registry.json"

    result = run_evolution(
        dataset=dataset,
        feature_pool=feat_pool,
        registry_path=registry_path,
        cfg=EvolutionConfig(
            generations=int(args.generations),
            max_models_per_generation=int(max_models),
            exploration_pct=float(evo_cfg.get("exploration_pct", 0.20)),
            seed=int(args.seed),
            budget=budget,
        ),
    )

    gen_rows: list[dict[str, Any]] = list(result.get("generations", []))
    pd.DataFrame(gen_rows).to_csv(out_dir / "generation_summary.csv", index=False)
    payload: dict[str, Any] = {
        "stage": "34.7",
        "run_id": run_id,
        "seed": int(args.seed),
        "budget": budget,
        "generation_count": int(args.generations),
        "max_models_per_generation": int(max_models),
        "did_generations_improve": bool(result.get("did_generations_improve", False)),
        "best_generation": int(result.get("best_generation", 0)),
        "generations": gen_rows,
        "config_hash": compute_config_hash(cfg),
        "data_hash": meta.get("data_hash", ""),
        "resolved_end_ts": meta.get("resolved_end_ts"),
        **snapshot_metadata_from_config(cfg),
    }
    (out_dir / "generation_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"generation_summary_json: {out_dir / 'generation_summary.json'}")
    print(f"generation_summary_csv: {out_dir / 'generation_summary.csv'}")


if __name__ == "__main__":
    main()
