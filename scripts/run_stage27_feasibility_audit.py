"""Stage-27.4 feasibility audit for execution rejects and sizing boundaries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.execution.feasibility import min_required_risk_pct
from buffmini.forensics.signal_flow import run_signal_flow_trace
from buffmini.stage24.sizing import cost_rt_pct_from_config
from buffmini.stage27.coverage_gate import evaluate_coverage_gate
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-27 feasibility audit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--allow-insufficient-data", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-27 Feasibility Report",
        "",
        f"- run_id: `{payload.get('run_id', '')}`",
        f"- seed: `{payload.get('seed', 0)}`",
        f"- dry_run: `{payload.get('dry_run', False)}`",
        f"- used_symbols: `{payload.get('used_symbols', [])}`",
        "",
        "## Top Reject Reasons",
    ]
    for item in payload.get("top_reject_reasons", [])[:10]:
        lines.append(f"- {item['reason']}: {item['count']}")

    lines.extend(
        [
            "",
            "## Feasible Percent by Equity Tier and Timeframe",
            "| timeframe | equity | feasible_pct | recommended_risk_floor |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in payload.get("feasibility_rows", []):
        lines.append(
            f"| {row['timeframe']} | {float(row['equity']):.2f} | {float(row['feasible_pct']):.6f} | {float(row['recommended_risk_floor']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- `minimum_required_risk_pct` is computed from stop distance + round-trip cost and min notional constraints.",
            "- High SIZE_TOO_SMALL/POLICY_CAP_HIT rates indicate feasibility bottlenecks, not alpha failure.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    started = time.perf_counter()
    symbols = _csv(args.symbols)
    timeframes = _csv(args.timeframes)
    stage26_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    gate = evaluate_coverage_gate(
        config=cfg,
        symbols=symbols,
        timeframe=str(stage26_cfg.get("base_timeframe", "1m")),
        data_dir=args.data_dir,
        allow_insufficient_data=bool(args.allow_insufficient_data),
        auto_btc_fallback=True,
    )
    if not gate.can_run:
        print(f"coverage_gate_status: {gate.status}")
        raise SystemExit(2)

    cfg_run = json.loads(json.dumps(cfg))
    cfg_run.setdefault("evaluation", {}).setdefault("stage23", {})["enabled"] = True
    cfg_run.setdefault("evaluation", {}).setdefault("stage24", {})["enabled"] = True
    cfg_run.setdefault("evaluation", {}).setdefault("stage24", {}).setdefault("sizing", {})["mode"] = "risk_pct"

    trace = run_signal_flow_trace(
        config=cfg_run,
        seed=int(args.seed),
        symbols=list(gate.used_symbols),
        timeframes=list(timeframes),
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
    reject_df = _read_csv(trace_dir / "execution_reject_events.csv")
    sizing_df = _read_csv(trace_dir / "sizing_trace.csv")

    top_reasons: list[dict[str, Any]] = []
    if not reject_df.empty:
        grouped = reject_df["reason"].astype(str).value_counts().head(10)
        top_reasons = [{"reason": str(k), "count": int(v)} for k, v in grouped.items()]

    cost_rt_pct = float(cost_rt_pct_from_config(cfg_run))
    stage24_cfg = dict(((cfg_run.get("evaluation", {}) or {}).get("stage24", {}) or {}))
    ladder_cfg = dict((stage24_cfg.get("sizing", {}) or {}).get("risk_ladder", {}))
    r_max = float(ladder_cfg.get("r_max", 0.20))
    max_notional_pct = float((stage24_cfg.get("order_constraints", {}) or {}).get("max_notional_pct_of_equity", 1.0))
    equity_tiers = [100.0, 1000.0, 10000.0, 100000.0]

    feasibility_rows: list[dict[str, Any]] = []
    if not sizing_df.empty:
        px = pd.to_numeric(sizing_df.get("price", 0.0), errors="coerce").fillna(0.0)
        stop_px = pd.to_numeric(sizing_df.get("stop_price", 0.0), errors="coerce").fillna(0.0)
        stop_dist = ((px - stop_px).abs() / px.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        min_notional = pd.to_numeric(sizing_df.get("min_notional", 0.0), errors="coerce").fillna(0.0)
        tf_col = sizing_df.get("timeframe", pd.Series(["unknown"] * len(sizing_df))).astype(str)

        for timeframe in sorted(set(tf_col.tolist())):
            mask = tf_col == str(timeframe)
            req_series = []
            for sd, mn in zip(stop_dist.loc[mask].tolist(), min_notional.loc[mask].tolist(), strict=False):
                req_series.append(
                    min_required_risk_pct(
                        equity=1000.0,
                        min_notional=float(mn),
                        stop_dist_pct=float(sd),
                        cost_rt_pct=float(cost_rt_pct),
                        max_notional_pct=float(max_notional_pct),
                    )
                )
            req = np.asarray(req_series, dtype=float)
            req = req[np.isfinite(req)]
            recommended_floor = float(np.quantile(req, 0.75)) if req.size else 0.0
            for equity in equity_tiers:
                feasible_flags = []
                for sd, mn in zip(stop_dist.loc[mask].tolist(), min_notional.loc[mask].tolist(), strict=False):
                    required = min_required_risk_pct(
                        equity=float(equity),
                        min_notional=float(mn),
                        stop_dist_pct=float(sd),
                        cost_rt_pct=float(cost_rt_pct),
                        max_notional_pct=float(max_notional_pct),
                    )
                    feasible_flags.append(bool(np.isfinite(required) and required <= float(r_max)))
                feasible_pct = float(np.mean(feasible_flags) * 100.0) if feasible_flags else 0.0
                feasibility_rows.append(
                    {
                        "timeframe": str(timeframe),
                        "equity": float(equity),
                        "feasible_pct": float(feasible_pct),
                        "recommended_risk_floor": float(recommended_floor),
                    }
                )

    payload = {
        "stage": "27.4",
        "run_id": str(trace["run_id"]),
        "seed": int(args.seed),
        "dry_run": bool(args.dry_run),
        "requested_symbols": list(gate.requested_symbols),
        "used_symbols": list(gate.used_symbols),
        "disabled_symbols": list(gate.disabled_symbols),
        "coverage_years_by_symbol": dict(gate.coverage_years_by_symbol),
        "top_reject_reasons": top_reasons,
        "feasibility_rows": feasibility_rows,
        "runtime_seconds": float(time.perf_counter() - started),
        "config_hash": compute_config_hash(cfg_run),
        "data_hash": str(trace.get("summary", {}).get("data_hash", "")),
        "resolved_end_ts": str(trace.get("summary", {}).get("resolved_end_ts", "")),
        "summary_hash": stable_hash({"reasons": top_reasons, "rows": feasibility_rows}, length=16),
    }

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / "stage27_feasibility_report.md"
    report_json = docs_dir / "stage27_feasibility_summary.json"
    report_md.write_text(_render_report(payload), encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")


if __name__ == "__main__":
    main()
