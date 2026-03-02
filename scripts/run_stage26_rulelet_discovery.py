"""Run Stage-26 rulelet discovery with conditional evaluation."""

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
from buffmini.stage26.rulelets import build_rulelet_library
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-26 rulelet discovery")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--timeframes", type=str, default="")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def _csv(value: str, default: list[str]) -> list[str]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or list(default)


def _cost_rows(cfg: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    stage12 = dict((cfg.get("evaluation", {}) or {}).get("stage12", {}))
    scenarios = dict(stage12.get("cost_scenarios", {}))
    costs = dict(cfg.get("costs", {}))
    base = {
        "round_trip_cost_pct": float(costs.get("round_trip_cost_pct", 0.1)),
        "slippage_pct": float(costs.get("slippage_pct", 0.0005)),
        "cost_model_cfg": cfg.get("cost_model", {}),
        "stop_atr_multiple": 1.5,
        "take_profit_atr_multiple": 3.0,
        "max_hold_bars": 24,
    }
    out = []
    for name in names:
        if str(name) == "realistic":
            row = dict(base)
            row["name"] = "realistic"
            out.append(row)
            continue
        s = dict(scenarios.get(str(name), {}))
        v2 = dict((cfg.get("cost_model", {}) or {}).get("v2", {}))
        if s:
            for k in ("slippage_bps_base", "slippage_bps_vol_mult", "spread_bps", "delay_bars"):
                if k in s and not bool(s.get("use_config_default", False)):
                    v2[k] = s[k]
        row = dict(base)
        row["name"] = str(name)
        row["cost_model_cfg"] = {**dict(cfg.get("cost_model", {})), "v2": v2}
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage26 = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    symbols = _csv(args.symbols, default=list(stage26.get("symbols", ["BTC/USDT", "ETH/USDT"])))
    timeframes = _csv(args.timeframes, default=list(stage26.get("timeframes", ["1h"])))
    cost_levels = list(stage26.get("cost_levels", ["realistic", "high"]))
    ctx_cfg = dict(stage26.get("context", {}))
    cond_cfg = dict(stage26.get("conditional_eval", {}))
    params = ContextParams(
        rank_window=int(ctx_cfg.get("rank_window", 252)),
        vol_window=int(ctx_cfg.get("vol_window", 24)),
        bb_window=int(ctx_cfg.get("bb_window", 20)),
        volume_window=int(ctx_cfg.get("volume_window", 120)),
        chop_window=int(ctx_cfg.get("chop_window", 48)),
        trend_lookback=int(ctx_cfg.get("trend_lookback", 24)),
    )
    cond_params = ConditionalEvalParams(
        bootstrap_samples=int(cond_cfg.get("bootstrap_samples", 500)),
        seed=int(args.seed),
        min_occurrences=int(cond_cfg.get("min_occurrences", 30)),
        min_trades=int(cond_cfg.get("min_trades", 30)),
        rare_min_trades=int(cond_cfg.get("rare_min_trades", 10)),
        rolling_months=tuple(int(v) for v in cond_cfg.get("rolling_months", [3, 6, 12])),
    )
    rulelets = build_rulelet_library()
    rows: list[dict[str, Any]] = []
    details_rows: list[dict[str, Any]] = []
    data_hashes: dict[str, str] = {}
    resolved_ends: list[pd.Timestamp] = []
    for tf in timeframes:
        frames = _build_features(
            config=cfg,
            symbols=symbols,
            timeframe=str(tf),
            dry_run=bool(args.dry_run),
            seed=int(args.seed),
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
        )
        for symbol, frame in frames.items():
            data_hashes[f"{symbol}|{tf}"] = _frame_data_hash(frame)
            ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
            if not ts.empty:
                resolved_ends.append(ts.max())
            with_ctx = classify_context(frame, params=params)
            effects, details = evaluate_rulelets_conditionally(
                frame=with_ctx,
                rulelets=rulelets,
                symbol=str(symbol),
                timeframe=str(tf),
                seed=int(args.seed),
                cost_levels=_cost_rows(cfg, cost_levels),
                params=cond_params,
            )
            if not effects.empty:
                rows.extend(effects.to_dict(orient="records"))
            details_rows.extend(list(details.get("rows", [])))

    out_df = pd.DataFrame(rows)
    run_id = f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'symbols': symbols, 'timeframes': timeframes, 'cfg': compute_config_hash(cfg), 'dry_run': bool(args.dry_run)}, length=12)}_stage26_rulelets"
    out_dir = args.runs_dir / run_id / "stage26"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "conditional_effects.csv", index=False)
    summary = {
        "stage": "26.4",
        "run_id": run_id,
        "seed": int(args.seed),
        "dry_run": bool(args.dry_run),
        "config_hash": compute_config_hash(cfg),
        "data_hash": stable_hash(data_hashes, length=16),
        "data_hashes_by_symbol_timeframe": data_hashes,
        "resolved_end_ts": max(resolved_ends).isoformat() if resolved_ends else None,
        "row_count": int(out_df.shape[0]),
        "class_counts": {str(k): int(v) for k, v in out_df.get("classification", pd.Series(dtype=str)).value_counts().items()},
        "rows": out_df.to_dict(orient="records"),
    }
    (out_dir / "conditional_effects.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    docs = Path("docs/stage26_rulelet_library.md")
    lines = [
        "# Stage-26 Rulelet Library",
        "",
        "## Catalog",
        "| rulelet | family | contexts_allowed | default_exit | threshold |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for item in rulelets.values():
        lines.append(
            f"| {item.name} | {item.family} | {','.join(item.contexts_allowed)} | {item.default_exit} | {float(item.threshold):.3f} |"
        )
    if not out_df.empty:
        lines.extend(
            [
                "",
                "## Conditional Summary",
                "| rulelet | context | trades_in_context | exp_lcb | classification |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        cols = ["rulelet", "context", "trades_in_context", "exp_lcb", "classification"]
        for row in out_df.sort_values(["rulelet", "context"]).loc[:, cols].to_dict(orient="records"):
            lines.append(
                f"| {row['rulelet']} | {row['context']} | {int(row['trades_in_context'])} | {float(row['exp_lcb']):.6f} | {row['classification']} |"
            )
    docs.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"run_id: {run_id}")
    print(f"conditional_effects_csv: {out_dir / 'conditional_effects.csv'}")
    print(f"conditional_effects_json: {out_dir / 'conditional_effects.json'}")
    print(f"docs: {docs}")


def _frame_data_hash(frame: pd.DataFrame) -> str:
    cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in frame.columns]
    if not cols:
        return stable_hash({"rows": int(frame.shape[0])}, length=16)
    payload = frame.loc[:, cols].to_dict(orient="list")
    return stable_hash(payload, length=16)


if __name__ == "__main__":
    main()
