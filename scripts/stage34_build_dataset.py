"""Stage-34.2 dataset builder runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage34.data_snapshot import audit_and_complete_snapshot
from buffmini.stage34.dataset_builder import DatasetConfig, build_stage34_dataset, write_stage34_dataset
from buffmini.stage34.features import feature_columns
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-34 supervised dataset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--timeframes", type=str, default="")
    parser.add_argument("--max-rows-per-symbol", type=int, default=300000)
    parser.add_argument("--max-features", type=int, default=120)
    parser.add_argument("--horizons-hours", type=str, default="24,72")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    text = str(value).strip()
    return [item.strip() for item in text.split(",") if item.strip()] if text else []


def _int_csv(value: str) -> list[int]:
    return [int(v) for v in _csv(value)]


def _resolve_symbols(cfg: dict[str, Any], symbols_arg: str) -> list[str]:
    cli = _csv(symbols_arg)
    if cli:
        return cli
    stage_cfg = ((cfg.get("evaluation", {}) or {}).get("stage34", {})) or {}
    symbols = [str(v) for v in stage_cfg.get("symbols", [])]
    return symbols or ["BTC/USDT", "ETH/USDT"]


def _resolve_timeframes(cfg: dict[str, Any], tf_arg: str) -> list[str]:
    cli = _csv(tf_arg)
    if cli:
        return cli
    stage_cfg = ((cfg.get("evaluation", {}) or {}).get("stage34", {})) or {}
    tfs = [str(v) for v in stage_cfg.get("timeframes", [])]
    return tfs or ["15m", "30m", "1h", "4h"]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage_cfg = ((cfg.get("evaluation", {}) or {}).get("stage34", {})) or {}
    symbols = _resolve_symbols(cfg, args.symbols)
    timeframes = _resolve_timeframes(cfg, args.timeframes)
    required_tfs = [str(v) for v in stage_cfg.get("required_timeframes", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])]
    horizons = _int_csv(args.horizons_hours) or [int(v) for v in (stage_cfg.get("dataset", {}) or {}).get("window_horizons_hours", [24, 72])]

    audit = audit_and_complete_snapshot(symbols=symbols, timeframes=required_tfs)
    resolved_end_ts = audit.get("resolved_end_ts")
    cfg_ds = DatasetConfig(
        symbols=tuple(symbols),
        timeframes=tuple(timeframes),
        max_rows_per_symbol=int(args.max_rows_per_symbol),
        max_features=int(args.max_features),
        horizons_hours=tuple(int(v) for v in horizons),
        resolved_end_ts=resolved_end_ts,
    )
    dataset, meta = build_stage34_dataset(cfg=cfg_ds)

    config_hash = compute_config_hash(cfg)
    data_hash = str(meta.get("data_hash", ""))
    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': int(args.seed), 'cfg': config_hash, 'data': data_hash, 'rows': int(meta.get('rows_total', 0)), 'resolved_end_ts': resolved_end_ts}, length=12)}"
        "_stage34_ds"
    )
    out_dir = Path(args.runs_dir) / run_id / "stage34" / "dataset"
    dataset_path, meta_path = write_stage34_dataset(dataset=dataset, meta=meta, out_dir=out_dir)

    docs_path = Path("docs/stage34_ml_dataset_spec.md")
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    spec = _render_dataset_spec(
        meta=meta,
        symbols=symbols,
        timeframes=timeframes,
        horizons=horizons,
        feature_count=int(len(feature_columns(int(args.max_features)))),
        config_hash=config_hash,
        data_hash=data_hash,
        resolved_end_ts=resolved_end_ts,
    )
    docs_path.write_text(spec, encoding="utf-8")

    summary = {
        "stage": "34.2",
        "run_id": run_id,
        "seed": int(args.seed),
        "dataset_path": str(dataset_path.as_posix()),
        "dataset_meta_path": str(meta_path.as_posix()),
        "config_hash": config_hash,
        "data_hash": data_hash,
        "resolved_end_ts": resolved_end_ts,
        "dataset_hash": str(meta.get("dataset_hash", "")),
        **snapshot_metadata_from_config(cfg),
    }
    (Path(args.runs_dir) / run_id / "stage34" / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"run_id: {run_id}")
    print(f"dataset: {dataset_path}")
    print(f"dataset_meta: {meta_path}")
    print(f"dataset_spec: {docs_path}")


def _render_dataset_spec(
    *,
    meta: dict[str, Any],
    symbols: list[str],
    timeframes: list[str],
    horizons: list[int],
    feature_count: int,
    config_hash: str,
    data_hash: str,
    resolved_end_ts: str | None,
) -> str:
    lines = [
        "# Stage-34 ML Dataset Spec",
        "",
        "## Scope",
        f"- symbols: `{symbols}`",
        f"- timeframes: `{timeframes}`",
        f"- horizons_hours: `{horizons}`",
        f"- rows_total: `{int(meta.get('rows_total', 0))}`",
        f"- max_features: `{int(meta.get('max_features', 0))}` (active={feature_count})",
        f"- max_rows_per_symbol: `{int(meta.get('max_rows_per_symbol', 0))}`",
        "",
        "## Feature List",
        *(f"- `{name}`" for name in meta.get("feature_columns", [])),
        "",
        "## Label Definitions",
        "- `label_primary`: primary triple-barrier-like direction label in {-1,0,1}.",
        "- `label_auxiliary`: forward adverse excursion proxy.",
        "",
        "## Row Counts",
    ]
    for key, value in sorted((meta.get("row_counts", {}) or {}).items()):
        lines.append(f"- `{key}`: `{int(value)}`")
    lines.extend(
        [
            "",
            "## Sampling / Limiting",
            "- Time-consistent truncation only (`tail(max_rows_per_symbol)`), no random sampling.",
            "",
            "## No-Leakage Guarantees",
            "- Features are computed from current/past bars only.",
            "- Labels may use forward horizons but are aligned to current timestamp.",
            "- Leakage harness tests include a synthetic intentionally leaky feature check.",
            "",
            "## Reproducibility",
            f"- config_hash: `{config_hash}`",
            f"- data_hash: `{data_hash}`",
            f"- dataset_hash: `{meta.get('dataset_hash', '')}`",
            f"- resolved_end_ts: `{resolved_end_ts}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    main()
