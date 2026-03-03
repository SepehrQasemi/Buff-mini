"""Run Stage-26 conditional policy replay in research/live mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage10.evaluate import _build_features
from buffmini.stage26.conditional_eval import ConditionalEvalParams, evaluate_rulelets_conditionally
from buffmini.stage26.context import ContextParams, classify_context
from buffmini.stage26.replay import replay_conditional_policy
from buffmini.stage26.rulelets import build_rulelet_library
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-26 policy replay")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", type=str, choices=["research", "live"], default="research")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--timeframes", type=str, default="")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _csv(value: str, default: list[str]) -> list[str]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or list(default)


def _prepare_frames(
    *,
    cfg: dict[str, Any],
    seed: int,
    dry_run: bool,
    symbols: list[str],
    timeframes: list[str],
    data_dir: Path,
    derived_dir: Path,
) -> dict[tuple[str, str], pd.DataFrame]:
    stage26 = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    ctx_cfg = dict(stage26.get("context", {}))
    params = ContextParams(
        rank_window=int(ctx_cfg.get("rank_window", 252)),
        vol_window=int(ctx_cfg.get("vol_window", 24)),
        bb_window=int(ctx_cfg.get("bb_window", 20)),
        volume_window=int(ctx_cfg.get("volume_window", 120)),
        chop_window=int(ctx_cfg.get("chop_window", 48)),
        trend_lookback=int(ctx_cfg.get("trend_lookback", 24)),
    )
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for tf in timeframes:
        frames = _build_features(
            config=cfg,
            symbols=symbols,
            timeframe=str(tf),
            dry_run=bool(dry_run),
            seed=int(seed),
            data_dir=data_dir,
            derived_dir=derived_dir,
        )
        for symbol, frame in frames.items():
            out[(str(symbol), str(tf))] = classify_context(frame, params=params)
    return out


def _effects_from_frames(
    *,
    frames: dict[tuple[str, str], pd.DataFrame],
    cfg: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    stage26 = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    cond_cfg = dict(stage26.get("conditional_eval", {}))
    params = ConditionalEvalParams(
        bootstrap_samples=int(cond_cfg.get("bootstrap_samples", 500)),
        seed=int(seed),
        min_occurrences=int(cond_cfg.get("min_occurrences", 30)),
        min_trades=int(cond_cfg.get("min_trades", 30)),
        rare_min_trades=int(cond_cfg.get("rare_min_trades", 10)),
        rolling_months=tuple(int(v) for v in cond_cfg.get("rolling_months", [3, 6, 12])),
    )
    costs = [{"name": "realistic", "round_trip_cost_pct": float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1)), "slippage_pct": float(cfg.get("costs", {}).get("slippage_pct", 0.0005)), "cost_model_cfg": cfg.get("cost_model", {})}]
    rulelets = build_rulelet_library()
    rows: list[dict[str, Any]] = []
    for (symbol, tf), frame in frames.items():
        table, _ = evaluate_rulelets_conditionally(
            frame=frame,
            rulelets=rulelets,
            symbol=symbol,
            timeframe=tf,
            seed=int(seed),
            cost_levels=costs,
            params=params,
        )
        if not table.empty:
            rows.extend(table.to_dict(orient="records"))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage26 = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    symbols = _csv(args.symbols, default=list(stage26.get("symbols", ["BTC/USDT", "ETH/USDT"])))
    timeframes = _csv(args.timeframes, default=list(stage26.get("timeframes", ["1h"])))
    mode = str(args.mode)
    constraints = cfg.setdefault("evaluation", {}).setdefault("constraints", {})
    if mode == "research":
        constraints["mode"] = str(stage26.get("constraints_mode_discovery", "research"))
    else:
        constraints["mode"] = str(stage26.get("constraints_mode_live", "live"))
    frames = _prepare_frames(
        cfg=cfg,
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        symbols=symbols,
        timeframes=timeframes,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    effects = _effects_from_frames(frames=frames, cfg=cfg, seed=int(args.seed))
    data_hashes = {f"{symbol}|{tf}": _frame_data_hash(frame) for (symbol, tf), frame in sorted(frames.items())}
    resolved_ends = [
        ts.max()
        for frame in frames.values()
        for ts in [pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()]
        if not ts.empty
    ]
    bundle = replay_conditional_policy(
        frames_by_symbol_tf=frames,
        effects=effects,
        config=cfg,
        seed=int(args.seed),
        mode=mode,
    )
    run_id = f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'symbols': symbols, 'timeframes': timeframes, 'mode': mode, 'dry_run': bool(args.dry_run), 'cfg': compute_config_hash(cfg)}, length=12)}_stage26_replay"
    out_dir = args.runs_dir / run_id / "stage26"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(bundle.metrics_rows)
    metrics_df.to_csv(out_dir / f"replay_{mode}.csv", index=False)
    bundle.policy_trace.to_csv(out_dir / "policy_trace.csv", index=False)
    shadow_df = pd.DataFrame(bundle.shadow_live_rows)
    if not shadow_df.empty:
        shadow_df.to_csv(out_dir / "shadow_live_rejects.csv", index=False)
    payload = {
        "stage": "26.7",
        "run_id": run_id,
        "mode": mode,
        "seed": int(args.seed),
        "dry_run": bool(args.dry_run),
        "config_hash": compute_config_hash(cfg),
        "data_hash": stable_hash(data_hashes, length=16),
        "data_hashes_by_symbol_timeframe": data_hashes,
        "resolved_end_ts": max(resolved_ends).isoformat() if resolved_ends else None,
        "metrics": bundle.metrics_rows,
        "policy": bundle.policy,
        "shadow_live_summary": _shadow_summary(shadow_df),
    }
    (out_dir / f"replay_{mode}.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs = docs_dir / "stage26_policy_composer.md"
    lines = [
        "# Stage-26 Policy Composer",
        "",
        f"- run_id: `{run_id}`",
        f"- mode: `{mode}`",
        f"- seed: `{int(args.seed)}`",
        "",
        "## Context Policies",
    ]
    contexts = dict(bundle.policy.get("contexts", {}))
    for ctx, item in sorted(contexts.items()):
        lines.append(f"- {ctx}:")
        lines.append(f"  - status: `{item.get('status', '')}`")
        lines.append(f"  - rulelets: `{item.get('rulelets', [])}`")
        lines.append(f"  - weights: `{item.get('weights', {})}`")
    lines.append("")
    lines.append("## Replay Metrics")
    lines.append("| symbol | timeframe | trade_count | exp_lcb | maxDD |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in metrics_df.to_dict(orient="records"):
        lines.append(
            f"| {row.get('symbol','')} | {row.get('timeframe','')} | {float(row.get('trade_count',0.0)):.2f} | {float(row.get('exp_lcb',0.0)):.6f} | {float(row.get('maxDD',0.0)):.6f} |"
        )
    docs.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"run_id: {run_id}")
    print(f"replay_csv: {out_dir / f'replay_{mode}.csv'}")
    print(f"replay_json: {out_dir / f'replay_{mode}.json'}")
    print(f"policy_trace: {out_dir / 'policy_trace.csv'}")
    if not shadow_df.empty:
        print(f"shadow_live: {out_dir / 'shadow_live_rejects.csv'}")
    print(f"docs: {docs}")


def _shadow_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "count": 0,
            "reject_rate": 0.0,
            "reasons": {},
        }
    reasons = df.get("live_reason", pd.Series(dtype=str)).astype(str).value_counts()
    rejected = int((df.get("live_reason", pd.Series(dtype=str)).astype(str) != "VALID").sum())
    return {
        "count": int(df.shape[0]),
        "reject_rate": float(rejected / max(1, int(df.shape[0]))),
        "reasons": {str(k): int(v) for k, v in reasons.items()},
    }


def _frame_data_hash(frame: pd.DataFrame) -> str:
    cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in frame.columns]
    if not cols:
        return stable_hash({"rows": int(frame.shape[0])}, length=16)
    payload = frame.loc[:, cols].to_dict(orient="list")
    return stable_hash(payload, length=16)


if __name__ == "__main__":
    main()
