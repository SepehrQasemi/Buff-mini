"""Run Stage-26 context classification audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage10.evaluate import _build_features
from buffmini.stage26.context import CONTEXTS, ContextParams, classify_context
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-26 context audit")
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


def _transition_table(states: pd.Series) -> pd.DataFrame:
    s = states.astype(str).reset_index(drop=True)
    trans = pd.DataFrame(0.0, index=CONTEXTS, columns=CONTEXTS)
    for a, b in zip(s.shift(1).iloc[1:], s.iloc[1:]):
        if str(a) in CONTEXTS and str(b) in CONTEXTS:
            trans.loc[str(a), str(b)] += 1.0
    rs = trans.sum(axis=1).replace(0.0, 1.0)
    return trans.div(rs, axis=0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage26 = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    symbols = _csv(args.symbols, default=list(stage26.get("symbols", ["BTC/USDT", "ETH/USDT"])))
    timeframes = _csv(args.timeframes, default=list(stage26.get("timeframes", ["1h"])))
    ctx_cfg = dict(stage26.get("context", {}))
    params = ContextParams(
        rank_window=int(ctx_cfg.get("rank_window", 252)),
        vol_window=int(ctx_cfg.get("vol_window", 24)),
        bb_window=int(ctx_cfg.get("bb_window", 20)),
        volume_window=int(ctx_cfg.get("volume_window", 120)),
        chop_window=int(ctx_cfg.get("chop_window", 48)),
        trend_lookback=int(ctx_cfg.get("trend_lookback", 24)),
    )
    rows: list[dict] = []
    transitions: list[dict] = []
    data_hashes: dict[str, str] = {}
    resolved_ends: list[pd.Timestamp] = []
    run_seed = int(args.seed)
    for tf in timeframes:
        frames = _build_features(
            config=cfg,
            symbols=symbols,
            timeframe=str(tf),
            dry_run=bool(args.dry_run),
            seed=run_seed,
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
        )
        for symbol, frame in frames.items():
            data_hashes[f"{symbol}|{tf}"] = _frame_data_hash(frame)
            ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
            if not ts.empty:
                resolved_ends.append(ts.max())
            with_ctx = classify_context(frame, params=params)
            counts = with_ctx["ctx_state"].astype(str).value_counts(normalize=True)
            for ctx in CONTEXTS:
                rows.append(
                    {
                        "symbol": str(symbol),
                        "timeframe": str(tf),
                        "context": str(ctx),
                        "pct": float(counts.get(ctx, 0.0) * 100.0),
                        "bars": int(with_ctx.shape[0]),
                    }
                )
            trans = _transition_table(with_ctx["ctx_state"])
            for src in CONTEXTS:
                for dst in CONTEXTS:
                    transitions.append(
                        {
                            "symbol": str(symbol),
                            "timeframe": str(tf),
                            "src": str(src),
                            "dst": str(dst),
                            "prob": float(trans.loc[src, dst]),
                        }
                    )
    run_id = f"{utc_now_compact()}_{stable_hash({'seed': run_seed, 'symbols': symbols, 'timeframes': timeframes, 'cfg': compute_config_hash(cfg)}, length=12)}_stage26_context"
    out_dir = args.runs_dir / run_id / "stage26"
    out_dir.mkdir(parents=True, exist_ok=True)
    dist_df = pd.DataFrame(rows)
    trans_df = pd.DataFrame(transitions)
    dist_df.to_csv(out_dir / "context_distribution.csv", index=False)
    trans_df.to_csv(out_dir / "context_transitions.csv", index=False)

    summary = {
        "stage": "26.2",
        "run_id": run_id,
        "seed": run_seed,
        "dry_run": bool(args.dry_run),
        "config_hash": compute_config_hash(cfg),
        "data_hash": stable_hash(data_hashes, length=16),
        "data_hashes_by_symbol_timeframe": data_hashes,
        "resolved_end_ts": max(resolved_ends).isoformat() if resolved_ends else None,
        "contexts": CONTEXTS,
        "distribution_rows": dist_df.to_dict(orient="records"),
    }
    (out_dir / "context_audit_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    docs = Path("docs/stage26_contexts.md")
    lines = [
        "# Stage-26 Contexts",
        "",
        f"- run_id: `{run_id}`",
        f"- seed: `{run_seed}`",
        f"- dry_run: `{bool(args.dry_run)}`",
        "",
        "| symbol | timeframe | context | pct | bars |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in dist_df.sort_values(["symbol", "timeframe", "context"]).to_dict(orient="records"):
        lines.append(
            f"| {row['symbol']} | {row['timeframe']} | {row['context']} | {float(row['pct']):.6f} | {int(row['bars'])} |"
        )
    docs.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"run_id: {run_id}")
    print(f"context_distribution: {out_dir / 'context_distribution.csv'}")
    print(f"context_transitions: {out_dir / 'context_transitions.csv'}")
    print(f"docs: {docs}")


def _frame_data_hash(frame: pd.DataFrame) -> str:
    cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in frame.columns]
    if not cols:
        return stable_hash({"rows": int(frame.shape[0])}, length=16)
    payload = frame.loc[:, cols].to_dict(orient="list")
    return stable_hash(payload, length=16)


if __name__ == "__main__":
    main()
