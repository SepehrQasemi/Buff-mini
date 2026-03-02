"""Stage-24 capital-level simulation helpers."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.forensics.signal_flow import run_signal_flow_trace
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def run_stage24_capital_sim(
    *,
    config: dict[str, Any],
    seed: int,
    dry_run: bool,
    symbols: list[str],
    base_timeframe: str,
    operational_timeframe: str,
    mode: str,
    initial_equities: list[float],
    runs_root: Path,
    data_dir: Path,
    derived_dir: Path,
    out_run_id: str | None = None,
    docs_dir: Path = Path("docs"),
) -> dict[str, Any]:
    """Run deterministic Stage-24 simulations across initial equity levels."""

    started = time.perf_counter()
    clean_mode = str(mode).strip().lower()
    if clean_mode not in {"risk_pct", "alloc_pct"}:
        raise ValueError("mode must be risk_pct|alloc_pct")
    equities = [float(x) for x in initial_equities if float(x) > 0.0]
    if not equities:
        raise ValueError("initial_equities must contain positive values")

    run_id = out_run_id or f"{utc_now_compact()}_{stable_hash({'seed': int(seed), 'mode': clean_mode, 'equities': equities, 'symbols': symbols, 'tf': operational_timeframe}, length=12)}_stage24_capital"
    run_dir = runs_root / run_id
    out_dir = run_dir / "stage24"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    trace_refs: list[dict[str, Any]] = []
    for equity in equities:
        cfg = deepcopy(config)
        cfg.setdefault("universe", {})["base_timeframe"] = str(base_timeframe)
        cfg.setdefault("data", {})["resample_source"] = "base"
        stage24 = cfg.setdefault("evaluation", {}).setdefault("stage24", {})
        stage24["enabled"] = True
        stage24["base_timeframe"] = str(base_timeframe)
        stage24["operational_timeframe"] = str(operational_timeframe)
        stage24.setdefault("sizing", {})
        stage24["sizing"]["mode"] = clean_mode
        stage24.setdefault("simulation", {})
        stage24["simulation"]["initial_equities"] = [float(equity)]
        stage24["simulation"]["seed"] = int(seed)

        trace = run_signal_flow_trace(
            config=cfg,
            seed=int(seed),
            symbols=list(symbols),
            timeframes=[str(operational_timeframe)],
            mode="classic",
            stages=["classic"],
            families=["price"],
            composers=["none"],
            max_combos=0,
            dry_run=bool(dry_run),
            runs_root=runs_root,
            data_dir=data_dir,
            derived_dir=derived_dir,
        )
        row, trace_ref = _collect_one_equity(
            initial_equity=float(equity),
            trace_result=trace,
            mode=clean_mode,
        )
        rows.append(row)
        trace_refs.append(trace_ref)

    results_df = pd.DataFrame(rows).sort_values("initial_equity").reset_index(drop=True)
    results_csv = out_dir / "capital_sim_results.csv"
    results_json = out_dir / "capital_sim_results.json"
    results_df.to_csv(results_csv, index=False)

    scale_invariance = _scale_invariance(results_df)
    payload = {
        "stage": "24.4",
        "run_id": run_id,
        "seed": int(seed),
        "mode": clean_mode,
        "dry_run": bool(dry_run),
        "base_timeframe": str(base_timeframe),
        "operational_timeframe": str(operational_timeframe),
        "symbols": list(symbols),
        "results_hash": _results_hash(results_df),
        "runtime_seconds": float(time.perf_counter() - started),
        "rows": results_df.to_dict(orient="records"),
        "scale_invariance_check": scale_invariance,
        "trace_refs": trace_refs,
    }
    results_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    docs_dir.mkdir(parents=True, exist_ok=True)
    capital_doc = docs_dir / "stage24_capital_sim.md"
    capital_doc.write_text(render_stage24_capital_sim_md(payload), encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "results_csv": results_csv,
        "results_json": results_json,
        "capital_doc": capital_doc,
        "summary": payload,
    }


def render_stage24_capital_sim_md(payload: dict[str, Any]) -> str:
    rows = list(payload.get("rows", []))
    scale = dict(payload.get("scale_invariance_check", {}))
    lines = [
        "# Stage-24 Capital Simulation",
        "",
        f"- run_id: `{payload.get('run_id', '')}`",
        f"- seed: `{payload.get('seed', 0)}`",
        f"- mode: `{payload.get('mode', '')}`",
        f"- dry_run: `{payload.get('dry_run', False)}`",
        f"- base_timeframe: `{payload.get('base_timeframe', '')}`",
        f"- operational_timeframe: `{payload.get('operational_timeframe', '')}`",
        f"- symbols: `{payload.get('symbols', [])}`",
        "",
        "## Results",
        "| initial_equity | final_equity | return_pct | max_drawdown | trade_count | avg_notional | avg_risk_pct_used | invalid_order_pct | top_invalid_reason |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {initial_equity:.2f} | {final_equity:.6f} | {return_pct:.6f} | {max_drawdown:.6f} | {trade_count:.2f} | {avg_notional:.6f} | {avg_risk_pct_used:.6f} | {invalid_order_pct:.6f} | {top_invalid_reason} |".format(
                **{
                    "initial_equity": float(row.get("initial_equity", 0.0)),
                    "final_equity": float(row.get("final_equity", 0.0)),
                    "return_pct": float(row.get("return_pct", 0.0)),
                    "max_drawdown": float(row.get("max_drawdown", 0.0)),
                    "trade_count": float(row.get("trade_count", 0.0)),
                    "avg_notional": float(row.get("avg_notional", 0.0)),
                    "avg_risk_pct_used": float(row.get("avg_risk_pct_used", 0.0)),
                    "invalid_order_pct": float(row.get("invalid_order_pct", 0.0)),
                    "top_invalid_reason": str(row.get("top_invalid_reason", "VALID")),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Scale Invariance Check",
            f"- return_pct_std: `{float(scale.get('return_pct_std', 0.0)):.6f}`",
            f"- scale_invariance_ok: `{bool(scale.get('scale_invariance_ok', False))}`",
            f"- note: `{scale.get('note', '')}`",
            "",
            "## Artifacts",
            f"- results_csv: `runs/{payload.get('run_id', '')}/stage24/capital_sim_results.csv`",
            f"- results_json: `runs/{payload.get('run_id', '')}/stage24/capital_sim_results.json`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _collect_one_equity(
    *,
    initial_equity: float,
    trace_result: dict[str, Any],
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = pd.DataFrame(trace_result.get("rows", pd.DataFrame()))
    summary = dict(trace_result.get("summary", {}))
    trace_dir = Path(trace_result.get("trace_dir", ""))
    breakdown = _read_json(trace_dir / "execution_reject_breakdown.json")
    s24_summary = _read_json(trace_dir / "stage24_sizing_summary.json")
    s24_trace = _read_csv(trace_dir / "stage24_sizing_trace.csv")
    sizing_summary = _read_json(trace_dir / "sizing_trace_summary.json")
    shadow_live_summary = _read_json(trace_dir / "shadow_live_summary.json")

    trade_count = float(pd.to_numeric(rows.get("trades_executed_count", 0.0), errors="coerce").fillna(0.0).sum()) if not rows.empty else 0.0
    trade_pnls = _collect_trade_pnls(rows)
    pnl_total = float(np.sum(trade_pnls)) if trade_pnls.size else 0.0
    final_equity = float(max(0.0, initial_equity + pnl_total))
    return_pct = float(((final_equity / initial_equity) - 1.0) * 100.0) if initial_equity > 0 else 0.0
    max_dd = float(pd.to_numeric(rows.get("maxDD", 0.0), errors="coerce").fillna(0.0).max()) if not rows.empty else 0.0

    attempted = float(breakdown.get("total_orders_attempted", 0.0))
    rejected = float(breakdown.get("total_orders_rejected", 0.0))
    invalid_order_pct = float((rejected / attempted) * 100.0) if attempted > 0 else 0.0
    top_invalid_reason = _top_reason(dict(breakdown.get("reject_reason_counts", {})))

    valid_mask = s24_trace.get("status", pd.Series(dtype=str)).astype(str) == "VALID" if not s24_trace.empty else pd.Series(dtype=bool)
    avg_notional = float(pd.to_numeric(s24_trace.loc[valid_mask, "notional"], errors="coerce").fillna(0.0).mean()) if not s24_trace.empty else 0.0
    avg_risk = float(pd.to_numeric(s24_trace.loc[valid_mask, "risk_used"], errors="coerce").fillna(0.0).mean()) if not s24_trace.empty else 0.0

    row = {
        "initial_equity": float(initial_equity),
        "mode": str(mode),
        "final_equity": final_equity,
        "return_pct": return_pct,
        "max_drawdown": max_dd,
        "trade_count": trade_count,
        "trades_per_month": float(pd.to_numeric(rows.get("tpm", 0.0), errors="coerce").fillna(0.0).sum()) if not rows.empty else 0.0,
        "avg_notional": avg_notional,
        "avg_risk_pct_used": avg_risk,
        "invalid_order_pct": invalid_order_pct,
        "top_invalid_reason": str(top_invalid_reason),
        "p_ruin": float(1.0 if final_equity <= 0.0 else 0.0),
        "walkforward_executed_true_pct": float(summary.get("walkforward_executed_true_pct", 0.0)),
        "mc_trigger_rate": float(summary.get("mc_trigger_rate", 0.0)),
        "config_hash": str(summary.get("config_hash", "")),
        "data_hash": str(summary.get("data_hash", "")),
        "resolved_end_ts": str(summary.get("resolved_end_ts", "")),
        "trace_run_id": str(trace_result.get("run_id", "")),
        "trace_dir": str(trace_dir),
        "stage24_valid_count": int(s24_summary.get("valid_count", 0)),
        "stage24_invalid_count": int(s24_summary.get("invalid_count", 0)),
        "margin_required_min": float(sizing_summary.get("margin_required_min", 0.0)),
        "margin_required_median": float(sizing_summary.get("margin_required_median", 0.0)),
        "margin_required_max": float(sizing_summary.get("margin_required_max", 0.0)),
        "margin_limit_min": float(sizing_summary.get("margin_limit_min", 0.0)),
        "margin_limit_median": float(sizing_summary.get("margin_limit_median", 0.0)),
        "margin_limit_max": float(sizing_summary.get("margin_limit_max", 0.0)),
        "cap_binding_reject_count": int(sizing_summary.get("cap_binding_reject_count", 0)),
        "reject_reason_counts": dict(breakdown.get("reject_reason_counts", {})),
        "research_accepted_but_live_rejected_count": int(shadow_live_summary.get("research_accepted_but_live_rejected_count", 0)),
        "shadow_live_reject_rate": float(shadow_live_summary.get("live_reject_rate", 0.0)),
    }
    trace_ref = {
        "initial_equity": float(initial_equity),
        "trace_run_id": str(trace_result.get("run_id", "")),
        "trace_dir": str(trace_dir),
    }
    return row, trace_ref


def _collect_trade_pnls(rows: pd.DataFrame) -> np.ndarray:
    if rows.empty or "trade_pnls" not in rows.columns:
        return np.asarray([], dtype=float)
    values: list[float] = []
    for item in rows["trade_pnls"].tolist():
        if isinstance(item, np.ndarray):
            arr = item.astype(float).tolist()
        elif isinstance(item, list):
            arr = item
        else:
            arr = []
        for val in arr:
            try:
                num = float(val)
            except Exception:
                continue
            if np.isfinite(num):
                values.append(num)
    if not values:
        return np.asarray([], dtype=float)
    return np.asarray(values, dtype=float)


def _top_reason(reason_counts: dict[str, Any]) -> str:
    pairs = [(str(reason), int(count)) for reason, count in reason_counts.items() if int(count) > 0 and str(reason) != "UNKNOWN"]
    if not pairs:
        if int(reason_counts.get("UNKNOWN", 0)) > 0:
            return "UNKNOWN"
        return "VALID"
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return pairs[0][0]


def _scale_invariance(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"return_pct_std": 0.0, "scale_invariance_ok": False, "note": "no_rows"}
    returns = pd.to_numeric(df.get("return_pct", 0.0), errors="coerce").fillna(0.0)
    std = float(returns.std(ddof=0))
    if len(returns) <= 1:
        return {"return_pct_std": std, "scale_invariance_ok": True, "note": "single_equity"}
    return {
        "return_pct_std": std,
        "scale_invariance_ok": bool(std <= 5.0),
        "note": "high dispersion suggests min_notional/cap effects" if std > 5.0 else "returns are broadly scale-consistent",
    }


def _results_hash(df: pd.DataFrame) -> str:
    if df.empty:
        return stable_hash([], length=16)
    keep = [
        "initial_equity",
        "mode",
        "final_equity",
        "return_pct",
        "max_drawdown",
        "trade_count",
        "trades_per_month",
        "avg_notional",
        "avg_risk_pct_used",
        "invalid_order_pct",
        "top_invalid_reason",
        "p_ruin",
        "walkforward_executed_true_pct",
        "mc_trigger_rate",
        "config_hash",
        "data_hash",
        "resolved_end_ts",
        "stage24_valid_count",
        "stage24_invalid_count",
    ]
    stable_df = df.loc[:, [col for col in keep if col in df.columns]].copy()
    stable_df = stable_df.sort_values("initial_equity").reset_index(drop=True)
    return stable_hash(stable_df.to_dict(orient="records"), length=16)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
