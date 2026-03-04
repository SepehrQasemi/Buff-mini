"""Run Stage-31 evolutionary synthesis orchestrator."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage10.evaluate import _build_features
from buffmini.stage31.evolve import EvolverConfig, evolve_strategies, signal_similarity
from buffmini.stage31.hyperband import HyperbandConfig, run_successive_halving
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-31 synthesis")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--population", type=int, default=120)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--elites", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _feature_list(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr_pct",
        "ema_20",
        "ema_50",
        "ema_200",
        "bb_bandwidth_20",
        "rsi_14",
        "volume_z_120",
    ]
    return [c for c in preferred if c in frame.columns] or [c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return ts.isoformat()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    symbols = _csv(args.symbols)
    timeframes = _csv(args.timeframes)
    seed = int(args.seed)
    config_hash = compute_config_hash(cfg)

    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': seed, 'symbols': symbols, 'timeframes': timeframes, 'pop': int(args.population), 'gen': int(args.generations), 'elite': int(args.elites), 'dry': bool(args.dry_run), 'cfg': config_hash}, length=12)}"
        "_stage31"
    )
    out_dir = Path(args.runs_dir) / run_id / "stage31"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    data_hash_parts: dict[str, str] = {}
    resolved_end: list[str] = []
    for timeframe in timeframes:
        feature_map = _build_features(
            config=cfg,
            symbols=symbols,
            timeframe=str(timeframe),
            dry_run=bool(args.dry_run),
            seed=seed,
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
        )
        for symbol, frame in sorted(feature_map.items()):
            features = _feature_list(frame)
            evolved = evolve_strategies(
                frame=frame,
                features=features,
                cfg=EvolverConfig(
                    population_size=int(args.population),
                    generations=int(args.generations),
                    elite_count=int(args.elites),
                    seed=seed,
                ),
            )
            halving = run_successive_halving(
                evolved.loc[:, ["strategy_id", "fitness"]].rename(columns={"fitness": "stage_a_score"}),
                cfg=HyperbandConfig(seed=seed),
            )
            top = evolved.copy()
            top["symbol"] = str(symbol)
            top["timeframe"] = str(timeframe)
            top["strategy_name"] = top["strategy"].map(lambda s: getattr(s, "name", ""))
            top["long_rule"] = top["strategy"].map(lambda s: getattr(s, "explain", lambda: {})().get("long_rule", ""))
            top["short_rule"] = top["strategy"].map(lambda s: getattr(s, "explain", lambda: {})().get("short_rule", ""))
            selected_ids = set()
            for frame_rung in halving.get("rungs", {}).values():
                selected_ids.update(str(v) for v in frame_rung.get("candidate_id", []).tolist())
            top["selected_by_halving"] = top["strategy_id"].astype(str).isin(selected_ids)

            data_cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in frame.columns]
            data_hash_parts[f"{symbol}|{timeframe}"] = stable_hash(frame.loc[:, data_cols].to_dict(orient="list"), length=16)
            ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
            if not ts.empty:
                resolved_end.append(ts.max().isoformat())
            rows.extend(top.to_dict(orient="records"))

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        candidates_top = all_df.copy()
    else:
        candidates_top = all_df.sort_values(["fitness", "strategy_id"], ascending=[False, True]).head(200).copy()
    candidates_top.to_csv(out_dir / "candidates_top.csv", index=False)

    records_json: list[dict[str, Any]] = []
    for rec in candidates_top.to_dict(orient="records"):
        rec_clean = {k: v for k, v in rec.items() if k not in {"strategy", "signal"}}
        records_json.append(_safe_json(rec_clean))
    (out_dir / "candidates_top.json").write_text(json.dumps(records_json, indent=2, allow_nan=False), encoding="utf-8")

    novelty_stats: dict[str, Any] = {"pair_count": 0, "mean_similarity": 0.0, "max_similarity": 0.0}
    if not candidates_top.empty:
        sims: list[float] = []
        sig_map = {
            str(row["strategy_id"]): row["signal"]
            for row in candidates_top.loc[:, ["strategy_id", "signal"]].to_dict(orient="records")
        }
        keys = list(sig_map.keys())[:50]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                sims.append(signal_similarity(sig_map[keys[i]], sig_map[keys[j]]))
        if sims:
            novelty_stats = {
                "pair_count": int(len(sims)),
                "mean_similarity": float(pd.Series(sims).mean()),
                "max_similarity": float(pd.Series(sims).max()),
            }
    (out_dir / "novelty_stats.json").write_text(json.dumps(novelty_stats, indent=2, allow_nan=False), encoding="utf-8")

    summary = {
        "stage": "31.4",
        "run_id": run_id,
        "seed": seed,
        "symbols": symbols,
        "timeframes": timeframes,
        "candidates_total": int(all_df.shape[0]),
        "candidates_top_count": int(candidates_top.shape[0]),
        "novelty_stats": novelty_stats,
        "config_hash": config_hash,
        "data_hash": stable_hash(data_hash_parts, length=16),
        "resolved_end_ts": max(resolved_end) if resolved_end else None,
        "runtime_seconds": float(time.perf_counter() - started),
        **snapshot_metadata_from_config(cfg),
    }
    (out_dir / "summary.json").write_text(json.dumps(_safe_json(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(f"run_id: {run_id}")
    print(f"stage31_dir: {out_dir}")


if __name__ == "__main__":
    main()

