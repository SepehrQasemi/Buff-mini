"""Run Stage-28 feasibility envelope analysis and write docs artifacts."""

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
from buffmini.forensics.signal_flow import run_signal_flow_trace
from buffmini.stage10.evaluate import _build_features
from buffmini.stage26.context import ContextParams, classify_context
from buffmini.stage24.sizing import cost_rt_pct_from_config
from buffmini.stage28.feasibility_envelope import compute_feasibility_envelope
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-28 feasibility envelope")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-28 Feasibility Envelope",
        "",
        f"- run_id: `{payload.get('run_id', '')}`",
        f"- seed: `{payload.get('seed', 42)}`",
        f"- dry_run: `{bool(payload.get('dry_run', False))}`",
        f"- symbols: `{payload.get('symbols', [])}`",
        f"- timeframes: `{payload.get('timeframes', [])}`",
        "",
        "## Shadow Live Rejections (Research Accepted But Live Rejected)",
        f"- count: `{int(payload.get('shadow_live_rejected_count', 0))}`",
        f"- rate: `{float(payload.get('shadow_live_reject_rate', 0.0)):.6f}`",
    ]
    reasons = list(payload.get("shadow_live_top_reasons", []))
    if reasons:
        lines.append("- top reasons:")
        for row in reasons[:10]:
            lines.append(f"  - {row.get('reason', '')}: {int(row.get('count', 0))}")

    lines.extend(
        [
            "",
            "## Feasibility Envelope by Equity Tier",
            "| symbol | timeframe | context | equity | feasible_pct | risk_p50 | risk_p90 | risk_floor |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("envelope_rows", [])[:200]:
        lines.append(
            "| {symbol} | {timeframe} | {context} | {equity:.0f} | {feasible_pct:.2f} | {min_required_risk_p50:.6f} | {min_required_risk_p90:.6f} | {recommended_risk_floor:.6f} |".format(
                **{
                    "symbol": row.get("symbol", ""),
                    "timeframe": row.get("timeframe", ""),
                    "context": row.get("context", ""),
                    "equity": float(row.get("equity", 0.0)),
                    "feasible_pct": float(row.get("feasible_pct", 0.0)),
                    "min_required_risk_p50": float(row.get("min_required_risk_p50", 0.0)),
                    "min_required_risk_p90": float(row.get("min_required_risk_p90", 0.0)),
                    "recommended_risk_floor": float(row.get("recommended_risk_floor", 0.0)),
                }
            )
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    symbols = _csv(args.symbols)
    timeframes = _csv(args.timeframes)
    seed = int(args.seed)

    cfg_run = json.loads(json.dumps(cfg))
    cfg_run.setdefault("evaluation", {}).setdefault("stage23", {})["enabled"] = True
    cfg_run.setdefault("evaluation", {}).setdefault("stage24", {})["enabled"] = True
    cfg_run.setdefault("evaluation", {}).setdefault("stage24", {}).setdefault("sizing", {})["mode"] = "risk_pct"
    cfg_run.setdefault("evaluation", {}).setdefault("constraints", {})["mode"] = "research"

    trace = run_signal_flow_trace(
        config=cfg_run,
        seed=seed,
        symbols=symbols,
        timeframes=timeframes,
        mode="classic",
        stages=["classic"],
        families=["price"],
        composers=["none"],
        max_combos=0,
        dry_run=bool(args.dry_run),
        runs_root=args.runs_dir,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    trace_dir = Path(trace["trace_dir"])
    stage24_trace = _read_csv(trace_dir / "stage24_sizing_trace.csv")
    shadow_live = _read_csv(trace_dir / "research_infeasible_flags.csv")

    stage26_cfg = dict((cfg_run.get("evaluation", {}) or {}).get("stage26", {})
    )
    ctx_cfg = dict(stage26_cfg.get("context", {}))
    ctx_params = ContextParams(
        rank_window=int(ctx_cfg.get("rank_window", 252)),
        vol_window=int(ctx_cfg.get("vol_window", 24)),
        bb_window=int(ctx_cfg.get("bb_window", 20)),
        volume_window=int(ctx_cfg.get("volume_window", 120)),
        chop_window=int(ctx_cfg.get("chop_window", 48)),
        trend_lookback=int(ctx_cfg.get("trend_lookback", 24)),
    )
    features_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for tf in timeframes:
        loaded = _build_features(
            config=cfg_run,
            symbols=symbols,
            timeframe=str(tf),
            dry_run=bool(args.dry_run),
            seed=seed,
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
        )
        for symbol, frame in sorted(loaded.items()):
            with_ctx = classify_context(frame, params=ctx_params)
            key = (str(symbol), str(tf))
            features_cache[key] = with_ctx

    if not stage24_trace.empty:
        stage24_trace["ts"] = pd.to_datetime(stage24_trace.get("ts"), utc=True, errors="coerce")
        stage24_trace["symbol"] = stage24_trace.get("symbol", "").astype(str)
        stage24_trace["timeframe"] = stage24_trace.get("timeframe", "").astype(str)
        context_values: list[str] = []
        for _, row in stage24_trace.iterrows():
            key = (str(row.get("symbol", "")), str(row.get("timeframe", "")))
            feat = features_cache.get(key)
            if feat is None or feat.empty:
                context_values.append("UNKNOWN")
                continue
            ts = pd.Timestamp(row.get("ts"))
            if pd.isna(ts):
                context_values.append("UNKNOWN")
                continue
            fts = pd.to_datetime(feat.get("timestamp"), utc=True, errors="coerce")
            match = feat.loc[fts == ts]
            if match.empty:
                context_values.append("UNKNOWN")
            else:
                context_values.append(str(match.iloc[-1].get("ctx_state", "UNKNOWN")))
        stage24_trace["context"] = context_values

    stage24_cfg = dict((cfg_run.get("evaluation", {}) or {}).get("stage24", {}))
    ladder_cfg = dict((stage24_cfg.get("sizing", {}) or {}).get("risk_ladder", {}))
    constraints_cfg = dict((stage24_cfg.get("order_constraints", {}) or {}))
    live_constraints = dict((((cfg_run.get("evaluation", {}) or {}).get("constraints", {}) or {}).get("live", {}) or {}))
    min_notional = float(live_constraints.get("min_trade_notional", constraints_cfg.get("min_trade_notional", 10.0)))
    max_notional_pct = float(constraints_cfg.get("max_notional_pct_of_equity", 1.0))
    risk_cap = float(ladder_cfg.get("r_max", 0.20))
    cost_rt = float(cost_rt_pct_from_config(cfg_run))
    equity_tiers = [100.0, 1000.0, 10000.0, 100000.0]

    envelope = compute_feasibility_envelope(
        signals=stage24_trace.loc[:, ["symbol", "timeframe", "context", "stop_dist_pct"]]
        if not stage24_trace.empty
        else pd.DataFrame(),
        equity_tiers=equity_tiers,
        min_notional=min_notional,
        cost_rt_pct=cost_rt,
        max_notional_pct=max_notional_pct,
        risk_cap=risk_cap,
    )

    shadow_reject_count = int(shadow_live.shape[0]) if not shadow_live.empty else 0
    shadow_total = int(max(1, int(stage24_trace.shape[0]) if not stage24_trace.empty else 1))
    shadow_rate = float(shadow_reject_count / shadow_total)
    top_reasons: list[dict[str, Any]] = []
    if not shadow_live.empty and "reason" in shadow_live.columns:
        grouped = shadow_live["reason"].astype(str).value_counts().head(10)
        top_reasons = [{"reason": str(name), "count": int(count)} for name, count in grouped.items()]

    run_id = str(trace.get("run_id", ""))
    out_dir = args.runs_dir / run_id / "stage28"
    out_dir.mkdir(parents=True, exist_ok=True)
    envelope_csv = out_dir / "feasibility_envelope.csv"
    envelope.to_csv(envelope_csv, index=False)

    payload = {
        "stage": "28.5",
        "run_id": run_id,
        "seed": seed,
        "dry_run": bool(args.dry_run),
        "symbols": list(symbols),
        "timeframes": list(timeframes),
        "equity_tiers": equity_tiers,
        "min_notional_live": float(min_notional),
        "risk_cap": float(risk_cap),
        "cost_rt_pct": float(cost_rt),
        "shadow_live_rejected_count": int(shadow_reject_count),
        "shadow_live_reject_rate": float(shadow_rate),
        "shadow_live_top_reasons": top_reasons,
        "envelope_rows": envelope.to_dict(orient="records"),
        "runtime_seconds": float(time.perf_counter() - started),
        "config_hash": compute_config_hash(cfg_run),
        "summary_hash": stable_hash(
            {
                "run_id": run_id,
                "rows": envelope.to_dict(orient="records"),
                "shadow": top_reasons,
                "seed": seed,
            },
            length=16,
        ),
        **snapshot_metadata_from_config(cfg_run),
    }

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / "stage28_feasibility_envelope.md"
    json_path = docs_dir / "stage28_feasibility_envelope.json"
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"trace_dir: {trace_dir}")
    print(f"envelope_csv: {envelope_csv}")
    print(f"report_md: {md_path}")
    print(f"report_json: {json_path}")


if __name__ == "__main__":
    main()
