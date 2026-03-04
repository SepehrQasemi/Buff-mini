
"""Stage-28 end-to-end orchestrator."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage10.evaluate import _build_features
from buffmini.stage26.conditional_eval import bootstrap_lcb
from buffmini.stage26.context import ContextParams, classify_context
from buffmini.stage26.rulelets import build_rulelet_library
from buffmini.stage27.coverage_gate import evaluate_coverage_gate
from buffmini.stage28.budget_funnel import BudgetFunnelConfig, run_budget_funnel
from buffmini.stage28.context_discovery import ContextCandidate, compute_context_signal, evaluate_context_candidate_matrix
from buffmini.stage28.feasibility_envelope import compute_feasibility_envelope
from buffmini.stage28.ml_ranker import MlRankerConfig, prioritize_candidates, train_ml_ranker
from buffmini.stage28.policy_v2 import PolicyV2Config, build_policy_v2, compose_policy_signal_v2, render_policy_spec_md
from buffmini.stage28.usability import UsabilityConfig, compute_usability
from buffmini.stage28.window_calendar import expected_window_count, generate_window_calendar
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-28 orchestrator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--timeframes", type=str, default="")
    parser.add_argument("--windows", type=str, default="3m,6m")
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--mode", type=str, default="research", choices=["research", "live", "both"])
    parser.add_argument("--enable-ml-ranker", action="store_true")
    parser.add_argument("--budget-small", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-insufficient-data", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _csv(value: str, default: list[str]) -> list[str]:
    out = [item.strip() for item in str(value).split(",") if item.strip()]
    return out or list(default)


def _months_tokens(value: str, default: list[int]) -> list[int]:
    tokens = _csv(value, [f"{item}m" for item in default])
    out: list[int] = []
    for token in tokens:
        text = str(token).strip().lower()
        if not text.endswith("m"):
            raise ValueError(f"Invalid month token: {token}")
        out.append(int(text[:-1]))
    return [int(item) for item in out if int(item) > 0]


def _context_params(stage26_cfg: dict[str, Any]) -> ContextParams:
    ctx = dict(stage26_cfg.get("context", {}))
    return ContextParams(
        rank_window=int(ctx.get("rank_window", 252)),
        vol_window=int(ctx.get("vol_window", 24)),
        bb_window=int(ctx.get("bb_window", 20)),
        volume_window=int(ctx.get("volume_window", 120)),
        chop_window=int(ctx.get("chop_window", 48)),
        trend_lookback=int(ctx.get("trend_lookback", 24)),
    )


def _cost_cfg(config: dict[str, Any]) -> dict[str, Any]:
    costs = dict(config.get("costs", {}))
    return {
        "round_trip_cost_pct": float(costs.get("round_trip_cost_pct", 0.1)),
        "slippage_pct": float(costs.get("slippage_pct", 0.0005)),
        "cost_model_cfg": config.get("cost_model", {}),
        "stop_atr_multiple": 1.5,
        "take_profit_atr_multiple": 3.0,
        "max_hold_bars": 24,
    }


def _policy_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "trade_count": 0.0,
            "tpm": 0.0,
            "PF_raw": 0.0,
            "PF_clipped": 0.0,
            "expectancy": 0.0,
            "exp_lcb": 0.0,
            "maxDD": 0.0,
            "zero_trade_pct": 100.0,
            "invalid_pct": 100.0,
        }
    frame = pd.DataFrame(rows)
    for col in ("trade_count", "tpm", "PF_raw", "PF_clipped", "expectancy", "exp_lcb", "maxDD"):
        frame[col] = pd.to_numeric(frame.get(col, 0.0), errors="coerce").fillna(0.0)
    invalid = (~np.isfinite(frame["exp_lcb"])) | (~np.isfinite(frame["PF_clipped"]))
    return {
        "trade_count": float(frame["trade_count"].sum()),
        "tpm": float(frame["tpm"].mean()),
        "PF_raw": float(frame["PF_raw"].mean()),
        "PF_clipped": float(frame["PF_clipped"].mean()),
        "expectancy": float(frame["expectancy"].mean()),
        "exp_lcb": float(frame["exp_lcb"].mean()),
        "maxDD": float(frame["maxDD"].mean()),
        "zero_trade_pct": float((frame["trade_count"] <= 0.0).mean() * 100.0),
        "invalid_pct": float(invalid.mean() * 100.0),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.tolist())
    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return ts.isoformat()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else 0.0
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _strategy_metrics(*, frame: pd.DataFrame, signal: pd.Series, symbol: str, timeframe: str, mode: str, cost_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    result = run_backtest(
        frame=frame.assign(signal=signal),
        strategy_name=f"Stage28Policy::{mode}",
        symbol=str(symbol),
        signal_col="signal",
        stop_atr_multiple=float(cost_cfg.get("stop_atr_multiple", 1.5)),
        take_profit_atr_multiple=float(cost_cfg.get("take_profit_atr_multiple", 3.0)),
        max_hold_bars=int(cost_cfg.get("max_hold_bars", 24)),
        round_trip_cost_pct=float(cost_cfg.get("round_trip_cost_pct", 0.1)),
        slippage_pct=float(cost_cfg.get("slippage_pct", 0.0005)),
        exit_mode="fixed_atr",
        cost_model_cfg=cost_cfg.get("cost_model_cfg", {}),
    )
    pnl = pd.to_numeric(result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    exp_lcb = bootstrap_lcb(values=pnl, seed=int(seed), samples=500)
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
    months = max(float((ts.iloc[-1] - ts.iloc[0]).total_seconds()) / 86400.0 / 30.0, 1e-9) if not ts.empty else 1e-9
    trade_count = float(result.metrics.get("trade_count", 0.0))
    pf_raw = float(result.metrics.get("profit_factor", 0.0))
    return {
        "symbol": str(symbol),
        "timeframe": str(timeframe),
        "mode": str(mode),
        "trade_count": float(trade_count),
        "tpm": float(trade_count / months),
        "exposure_ratio": float((signal != 0).mean()),
        "PF_raw": float(pf_raw),
        "PF_clipped": float(np.clip(pf_raw if np.isfinite(pf_raw) else 0.0, 0.0, 10.0)),
        "expectancy": float(result.metrics.get("expectancy", 0.0)),
        "exp_lcb": float(exp_lcb),
        "maxDD": float(result.metrics.get("max_drawdown", 0.0)),
    }


def _apply_live_constraints(signal: pd.Series, trace: pd.DataFrame, close: pd.Series, live_cfg: dict[str, Any]) -> tuple[pd.Series, list[dict[str, Any]]]:
    filtered = pd.to_numeric(signal, errors="coerce").fillna(0).astype(int).copy()
    min_notional = float(live_cfg.get("min_trade_notional", 10.0))
    min_qty = float(live_cfg.get("min_trade_qty", 0.0))
    qty_step = float(live_cfg.get("qty_step", 0.0))
    rejects: list[dict[str, Any]] = []
    for idx, row in trace.iterrows():
        if idx >= len(filtered):
            continue
        sig = int(row.get("final_signal", 0))
        if sig == 0:
            continue
        qty = abs(float(row.get("net_score", 0.0)))
        notional = float(qty * close.iloc[idx]) if idx < len(close) else 0.0
        reason = "VALID"
        if qty <= 0.0:
            reason = "SIZE_ZERO"
        elif qty < min_qty:
            reason = "SIZE_TOO_SMALL"
        elif qty_step > 0 and abs((qty / qty_step) - round(qty / qty_step)) > 1e-9:
            reason = "SIZE_TOO_SMALL"
        elif notional < min_notional:
            reason = "SIZE_TOO_SMALL"
        if reason != "VALID":
            filtered.iloc[idx] = 0
            rejects.append({"timestamp": str(row.get("timestamp", "")), "context": str(row.get("context", "")), "reason": reason, "qty_proxy": qty, "notional_proxy": notional})
    return filtered, rejects

def _top_edges(frame: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    ranked = frame.sort_values(["exp_lcb", "trades_in_context", "context_occurrences", "candidate_id"], ascending=[False, False, False, True]).head(int(limit))
    cols = [
        "candidate_id",
        "candidate",
        "context",
        "symbol",
        "timeframe",
        "exp_lcb",
        "trades_in_context",
        "context_occurrences",
        "classification",
    ]
    return ranked.loc[:, [c for c in cols if c in ranked.columns]].to_dict(orient="records")


def _render_master(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-28 Master Report",
        "",
        f"- run_id: `{payload.get('run_id', '')}`",
        f"- mode: `{payload.get('mode', '')}`",
        f"- dry_run: `{payload.get('dry_run', False)}`",
        f"- data_snapshot_id: `{payload.get('data_snapshot_id', '')}`",
        f"- data_snapshot_hash: `{payload.get('data_snapshot_hash', '')}`",
        "",
        "## Window Counts",
    ]
    for key, item in sorted(dict(payload.get("window_counts", {})).items(), key=lambda kv: int(kv[0])):
        lines.append(f"- {key}m: generated={int(item.get('generated', 0))}, evaluated={int(item.get('evaluated', 0))}, expected={int(item.get('expected', 0))}")
    lines.extend(
        [
            "",
            f"- wf_executed_pct: `{float(payload.get('wf_executed_pct', 0.0)):.6f}`",
            f"- mc_trigger_pct: `{float(payload.get('mc_trigger_pct', 0.0)):.6f}`",
            "",
            "## Policy Metrics",
            f"- research: `{payload.get('policy_metrics', {}).get('research', {})}`",
            f"- live: `{payload.get('policy_metrics', {}).get('live', {})}`",
            "",
            "## Feasibility",
            f"- shadow_live_reject_rate: `{float(payload.get('shadow_live_reject_rate', 0.0)):.6f}`",
            f"- avg_feasible_pct_by_equity: `{payload.get('feasibility_summary', {}).get('avg_feasible_pct_by_equity', {})}`",
            "",
            "## Top Contextual Edges",
        ]
    )
    for row in payload.get("top_contextual_edges", [])[:10]:
        lines.append(
            "- {candidate} | {context} | {symbol}/{timeframe} | exp_lcb={exp_lcb:.6f} | trades={trades_in_context} | occ={context_occurrences}".format(
                candidate=str(row.get("candidate", row.get("candidate_id", ""))),
                context=str(row.get("context", "")),
                symbol=str(row.get("symbol", "")),
                timeframe=str(row.get("timeframe", "")),
                exp_lcb=float(row.get("exp_lcb", 0.0)),
                trades_in_context=int(row.get("trades_in_context", 0)),
                context_occurrences=int(row.get("context_occurrences", 0)),
            )
        )
    lines.extend(["", "## Verdict", f"- `{payload.get('verdict', '')}`", f"- next_bottleneck: `{payload.get('next_bottleneck', '')}`"])
    return "\n".join(lines).strip() + "\n"


def _render_product_spec(payload: dict[str, Any]) -> str:
    return (
        "# Stage-28 Product Spec\n\n"
        "Buff-mini Edge Engine (Local) outputs a contextual policy, evidence tables, and live feasibility envelope.\n\n"
        f"- symbols: `{payload.get('used_symbols', [])}`\n"
        f"- timeframes: `{payload.get('timeframes', [])}`\n"
        f"- window_months: `{payload.get('window_months', [])}`\n"
        f"- verdict: `{payload.get('verdict', '')}`\n"
    )


def _git_head() -> str:
    head = Path(".git/HEAD")
    if not head.exists():
        return ""
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = Path(".git") / text.split(" ", 1)[1].strip()
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
    return text


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    stage26_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    stage28_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage28", {}))
    seed = int(args.seed)
    mode = str(args.mode).strip().lower()
    dry_run = bool(args.dry_run)
    symbols = _csv(args.symbols, list(stage28_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])))
    timeframes = _csv(args.timeframes, list(stage28_cfg.get("timeframes", ["15m", "30m", "1h", "2h", "4h"])))
    window_months = _months_tokens(args.windows, [int(v) for v in stage28_cfg.get("windows", [3, 6])])
    step_months = int(max(1, args.step_months if args.step_months else int(stage28_cfg.get("step_months", 1))))
    snapshot_meta = snapshot_metadata_from_config(cfg)
    config_hash = compute_config_hash(cfg)

    used_symbols = list(symbols)
    coverage_status = "SKIPPED_DRY_RUN"
    coverage_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not dry_run:
        gate = evaluate_coverage_gate(
            config=cfg,
            symbols=symbols,
            timeframe=str(stage26_cfg.get("base_timeframe", "1m")),
            data_dir=args.data_dir,
            allow_insufficient_data=bool(args.allow_insufficient_data),
            auto_btc_fallback=True,
        )
        coverage_status = str(gate.status)
        coverage_rows = list(gate.rows)
        if not gate.can_run:
            payload = {
                "stage": "28",
                "run_id": "",
                "seed": seed,
                "mode": mode,
                "dry_run": dry_run,
                "head_commit": _git_head(),
                "config_hash": config_hash,
                "data_hash": "",
                "resolved_end_ts": None,
                "coverage_gate_status": coverage_status,
                "coverage_rows": coverage_rows,
                "requested_symbols": list(gate.requested_symbols),
                "used_symbols": list(gate.used_symbols),
                "timeframes": timeframes,
                "window_months": window_months,
                "step_months": step_months,
                "window_counts": {str(m): {"generated": 0, "evaluated": 0, "expected": 0} for m in window_months},
                "funnel_summary": {},
                "wf_executed_pct": 0.0,
                "mc_trigger_pct": 0.0,
                "qualified_finalists": 0,
                "top_contextual_edges": [],
                "policy_metrics": {"research": _policy_metrics([]), "live": _policy_metrics([])},
                "shadow_live_reject_rate": 0.0,
                "shadow_live_top_reasons": {},
                "feasibility_summary": {"equity_tiers": [100, 1000, 10000, 100000], "avg_feasible_pct_by_equity": {}},
                "verdict": "INSUFFICIENT_DATA",
                "next_bottleneck": "data_coverage",
                "warnings": [],
                "runtime_seconds": float(time.perf_counter() - started),
                **snapshot_meta,
            }
            docs = Path(args.docs_dir)
            docs.mkdir(parents=True, exist_ok=True)
            (docs / "stage28_master_summary.json").write_text(
                json.dumps(_json_safe(payload), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            (docs / "stage28_master_report.md").write_text(_render_master(payload), encoding="utf-8")
            (docs / "stage28_product_spec.md").write_text(_render_product_spec(payload), encoding="utf-8")
            raise SystemExit(2)
        used_symbols = list(gate.used_symbols)
        warnings.extend([str(note) for note in gate.notes])
    funnel_cfg = dict(stage28_cfg.get("funnel", {}))
    if bool(args.budget_small):
        funnel_cfg["stage_b_budget"] = min(int(funnel_cfg.get("stage_b_budget", 60)), 20)
        funnel_cfg["stage_c_budget"] = min(int(funnel_cfg.get("stage_c_budget", 25)), 8)

    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': seed, 'symbols': used_symbols, 'timeframes': timeframes, 'windows': window_months, 'step_months': step_months, 'mode': mode, 'dry': dry_run, 'cfg': config_hash, 'snap': snapshot_meta.get('data_snapshot_hash', '')}, length=12)}"
        "_stage28"
    )
    run_dir = Path(args.runs_dir) / run_id / "stage28"
    run_dir.mkdir(parents=True, exist_ok=True)
    docs = Path(args.docs_dir)
    docs.mkdir(parents=True, exist_ok=True)

    context_params = _context_params(stage26_cfg)
    rulelet_library = build_rulelet_library()
    cost_cfg = _cost_cfg(cfg)
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    data_hash_parts: dict[str, str] = {}
    resolved_end: list[str] = []

    for tf in timeframes:
        feature_map = _build_features(
            config=cfg,
            symbols=used_symbols,
            timeframe=str(tf),
            dry_run=dry_run,
            seed=seed,
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
        )
        for symbol, frame in sorted(feature_map.items()):
            ctx_frame = classify_context(frame, params=context_params)
            frames[(str(symbol), str(tf))] = ctx_frame
            cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in ctx_frame.columns]
            data_hash_parts[f"{symbol}|{tf}"] = stable_hash(ctx_frame[cols].to_dict(orient="list"), length=16)
            ts = pd.to_datetime(ctx_frame.get("timestamp"), utc=True, errors="coerce").dropna()
            if not ts.empty:
                resolved_end.append(ts.max().isoformat())

    calendar_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    window_counts = {str(m): {"generated": 0, "evaluated": 0, "expected": 0} for m in window_months}

    for (symbol, tf), frame in sorted(frames.items()):
        ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
        if ts.empty:
            continue
        for months in window_months:
            expected = expected_window_count(start_ts=ts.iloc[0], end_ts=ts.iloc[-1], window_months=int(months), step_months=int(step_months))
            calendar = generate_window_calendar(ts, window_months=int(months), step_months=int(step_months))
            window_counts[str(months)]["generated"] += int(calendar.shape[0])
            window_counts[str(months)]["expected"] += int(expected)
            for row in calendar.to_dict(orient="records"):
                row.update({"symbol": str(symbol), "timeframe": str(tf), "window_months": int(months)})
                calendar_rows.append(row)
                start_ts = pd.to_datetime(row["window_start"], utc=True, errors="coerce")
                end_ts = pd.to_datetime(row["window_end"], utc=True, errors="coerce")
                if pd.isna(start_ts) or pd.isna(end_ts):
                    continue
                mask = (pd.to_datetime(frame["timestamp"], utc=True, errors="coerce") >= start_ts) & (pd.to_datetime(frame["timestamp"], utc=True, errors="coerce") < end_ts)
                sliced = frame.loc[mask].reset_index(drop=True)
                if sliced.shape[0] < 300:
                    continue
                window_counts[str(months)]["evaluated"] += 1
                matrix = evaluate_context_candidate_matrix(frame=sliced, symbol=str(symbol), timeframe=str(tf), seed=seed, rulelet_library=rulelet_library)
                if matrix.empty:
                    continue
                matrix = matrix.copy()
                matrix["window_months"] = int(months)
                matrix["window_index"] = int(row["window_index"])
                matrix["window_start"] = str(row["window_start"])
                matrix["window_end"] = str(row["window_end"])
                matrix_rows.extend(matrix.to_dict(orient="records"))

    calendar_df = pd.DataFrame(calendar_rows)
    matrix_df = pd.DataFrame(matrix_rows)
    calendar_df.to_csv(run_dir / "window_calendar.csv", index=False)
    matrix_df.to_csv(run_dir / "context_candidate_matrix.csv", index=False)
    (run_dir / "context_candidate_matrix.json").write_text(
        json.dumps(_json_safe({"rows": matrix_df.to_dict(orient="records")}), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    funnel = run_budget_funnel(
        candidates=matrix_df,
        signal_map={},
        cfg=BudgetFunnelConfig(
            stage_b_top_pct=float(funnel_cfg.get("stage_b_top_pct", 0.35)),
            stage_b_budget=int(funnel_cfg.get("stage_b_budget", 60)),
            stage_c_budget=int(funnel_cfg.get("stage_c_budget", 25)),
            exploration_pct=float(funnel_cfg.get("exploration_pct", 0.15)),
            min_exploration_pct=float(funnel_cfg.get("min_exploration_pct", 0.10)),
            sim_threshold=float(funnel_cfg.get("sim_threshold", 0.90)),
            seed=seed,
        ),
    )
    stage_a = funnel["stage_a"].copy()
    stage_b = funnel["stage_b"].copy()
    stage_c = funnel["stage_c"].copy()

    ml_cfg_raw = dict(stage28_cfg.get("ml_ranker", {}))
    ml_enabled = bool(args.enable_ml_ranker) or bool(ml_cfg_raw.get("enabled", False))
    ml_info: dict[str, Any] = {"enabled": ml_enabled}
    if ml_enabled:
        ml_cfg = MlRankerConfig(
            enabled=True,
            exploration_pct=float(ml_cfg_raw.get("exploration_pct", 0.15)),
            min_exploration_pct=float(funnel_cfg.get("min_exploration_pct", 0.10)),
            alpha=float(ml_cfg_raw.get("alpha", 1.0)),
            max_features=int(ml_cfg_raw.get("max_features", 20)),
            seed=seed,
        )
        model = train_ml_ranker(stage_a, cfg=ml_cfg)
        stage_b_budget = int(max(1, min(int(funnel_cfg.get("stage_b_budget", 60)), stage_a.shape[0] if not stage_a.empty else 1)))
        stage_b = prioritize_candidates(stage_a, budget=stage_b_budget, model=model, cfg=ml_cfg)
        stage_c_budget = int(max(1, int(funnel_cfg.get("stage_c_budget", 25))))
        stage_c = stage_b.sort_values(["ml_score", "candidate_id"], ascending=[False, True]).head(stage_c_budget).copy()
        if "stage_c_score" not in stage_c.columns:
            stage_c["stage_c_score"] = pd.to_numeric(stage_c.get("ml_score", 0.0), errors="coerce").fillna(0.0)
        ml_info.update({"stage_b_budget": stage_b_budget, "stage_c_budget": stage_c_budget})

    stage_a.to_csv(run_dir / "selected_candidates_stageA.csv", index=False)
    stage_b.to_csv(run_dir / "selected_candidates_stageB.csv", index=False)
    stage_c.to_csv(run_dir / "finalists_stageC.csv", index=False)
    (run_dir / "funnel_summary.json").write_text(
        json.dumps(_json_safe({"summary": funnel.get("summary", {}), "ml": ml_info}), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    usability_rows: list[dict[str, Any]] = []
    for row in stage_c.to_dict(orient="records"):
        cid = str(row.get("candidate_id", ""))
        if not cid:
            continue
        subset = matrix_df.loc[matrix_df.get("candidate_id", "").astype(str) == cid, :].copy()
        usage = compute_usability(
            candidate={
                "context_occurrences": float(pd.to_numeric(subset.get("context_occurrences", 0), errors="coerce").fillna(0.0).sum()),
                "trades_in_context": float(pd.to_numeric(subset.get("trades_in_context", 0), errors="coerce").fillna(0.0).sum()),
            },
            windows=pd.DataFrame({
                "window_id": subset.get("window_index", pd.Series(dtype=int)),
                "trade_count": pd.to_numeric(subset.get("trades_in_context", 0), errors="coerce").fillna(0.0),
                "occurrences": pd.to_numeric(subset.get("context_occurrences", 0), errors="coerce").fillna(0.0),
            }),
            cfg=UsabilityConfig(min_trades_context=30, min_occurrences_context=50, min_windows=3, rare_pool_min_trades=30),
        )
        usability_rows.append({"candidate_id": cid, "context": str(row.get("context", "")), "symbol": str(row.get("symbol", "")), "timeframe": str(row.get("timeframe", "")), **usage})
    usability_df = pd.DataFrame(usability_rows)
    usability_df.to_csv(run_dir / "usability_trace.csv", index=False)
    wf_pct = float(pd.to_numeric(usability_df.get("wf_triggered", False), errors="coerce").fillna(0).astype(bool).mean() * 100.0) if not usability_df.empty else 0.0
    mc_pct = float(pd.to_numeric(usability_df.get("mc_triggered", False), errors="coerce").fillna(0).astype(bool).mean() * 100.0) if not usability_df.empty else 0.0

    policy = build_policy_v2(
        stage_c,
        data_snapshot_id=str(snapshot_meta.get("data_snapshot_id", "")),
        data_snapshot_hash=str(snapshot_meta.get("data_snapshot_hash", "")),
        config_hash=str(config_hash),
        cfg=PolicyV2Config(top_k_per_context=3 if not args.budget_small else 2, min_occurrences_context=50, min_trades_context=30, min_exp_lcb=0.0, w_min=0.05, w_max=0.80, conflict_mode="net"),
    )
    (run_dir / "policy.json").write_text(json.dumps(_json_safe(policy), indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "policy_spec.md").write_text(render_policy_spec_md(policy), encoding="utf-8")

    live_cfg = dict(((cfg.get("evaluation", {}) or {}).get("constraints", {}) or {}).get("live", {}))
    metrics_research_rows: list[dict[str, Any]] = []
    metrics_live_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    feasibility_rows: list[dict[str, Any]] = []

    for (symbol, tf), frame in sorted(frames.items()):
        local = stage_c.loc[(stage_c.get("symbol", "").astype(str) == str(symbol)) & (stage_c.get("timeframe", "").astype(str) == str(tf))].copy()
        if local.empty:
            continue
        sig_map: dict[str, pd.Series] = {}
        for row in local.to_dict(orient="records"):
            name = str(row.get("candidate", ""))
            if name not in rulelet_library:
                continue
            rulelet = rulelet_library[name]
            candidate = ContextCandidate(name=name, family=str(row.get("family", getattr(rulelet, "family", "unknown"))), context=str(row.get("context", "")), threshold=float(getattr(rulelet, "threshold", 0.30)), default_exit=str(getattr(rulelet, "default_exit", "fixed_atr")), required_features=tuple(str(c) for c in rulelet.required_features()))
            _, sig, _ = compute_context_signal(frame=frame, candidate=candidate, rulelet_library=rulelet_library, shift_entries=True)
            sig_map[str(row.get("candidate_id", ""))] = sig
        if not sig_map:
            continue
        signal_research, trace = compose_policy_signal_v2(frame=frame, policy=policy, candidate_signals=sig_map)
        trace = trace.copy()
        trace["symbol"] = str(symbol)
        trace["timeframe"] = str(tf)
        trace_rows.extend(trace.to_dict(orient="records"))

        metrics_research_rows.append(_strategy_metrics(frame=frame, signal=signal_research, symbol=str(symbol), timeframe=str(tf), mode="research", cost_cfg=cost_cfg, seed=seed))

        close = pd.to_numeric(frame.get("close", 0.0), errors="coerce").fillna(0.0)
        signal_live, rejects = _apply_live_constraints(signal_research, trace, close, live_cfg)
        for reject in rejects:
            reject["symbol"] = str(symbol)
            reject["timeframe"] = str(tf)
        shadow_rows.extend(rejects)

        metrics_live_rows.append(_strategy_metrics(frame=frame, signal=signal_live if mode in {"live", "both"} else signal_research, symbol=str(symbol), timeframe=str(tf), mode="live", cost_cfg=cost_cfg, seed=seed + 1))

        atr_pct = pd.to_numeric(frame.get("atr_pct", 0.0), errors="coerce").fillna(0.0)
        for idx, item in trace.iterrows():
            if int(item.get("final_signal", 0)) == 0:
                continue
            stop_pct = float(abs(atr_pct.iloc[idx])) if idx < len(atr_pct) else 0.0
            feasibility_rows.append({"symbol": str(symbol), "timeframe": str(tf), "context": str(item.get("context", "")), "stop_dist_pct": stop_pct if stop_pct > 0 else 0.01})

    trace_df = pd.DataFrame(trace_rows)
    shadow_df = pd.DataFrame(shadow_rows)
    trace_df.to_csv(run_dir / "policy_trace.csv", index=False)
    shadow_df.to_csv(run_dir / "shadow_live_rejects.csv", index=False)

    research_metrics = _policy_metrics(metrics_research_rows)
    live_metrics = _policy_metrics(metrics_live_rows)
    nonzero_signals = int((trace_df.get("final_signal", pd.Series(dtype=int)) != 0).sum()) if not trace_df.empty else 0
    shadow_rate = float(shadow_df.shape[0] / max(1, nonzero_signals))
    top_reasons = shadow_df["reason"].value_counts().head(5).to_dict() if ("reason" in shadow_df.columns and not shadow_df.empty) else {}

    envelope_df = compute_feasibility_envelope(
        signals=pd.DataFrame(feasibility_rows),
        equity_tiers=[100.0, 1000.0, 10000.0, 100000.0],
        min_notional=float(live_cfg.get("min_trade_notional", 10.0)),
        cost_rt_pct=float(cost_cfg.get("round_trip_cost_pct", 0.1)),
        max_notional_pct=float((((cfg.get("evaluation", {}) or {}).get("stage24", {}) or {}).get("order_constraints", {}) or {}).get("max_notional_pct_of_equity", 1.0)),
        risk_cap=float((((cfg.get("evaluation", {}) or {}).get("stage24", {}) or {}).get("sizing", {}) or {}).get("risk_ladder", {}).get("r_max", 0.20)),
    )
    envelope_df.to_csv(run_dir / "feasibility_envelope.csv", index=False)
    avg_feasible = {}
    if not envelope_df.empty:
        grouped = envelope_df.groupby("equity", dropna=False)["feasible_pct"].mean().to_dict()
        avg_feasible = {str(int(k)): float(v) for k, v in grouped.items()}

    top_edges = _top_edges(stage_c, limit=10)
    if float(live_metrics.get("exp_lcb", 0.0)) > 0 and wf_pct >= 60.0 and mc_pct >= 30.0:
        verdict = "ROBUST_EDGE"
    elif float(live_metrics.get("exp_lcb", 0.0)) > 0 and wf_pct > 0.0:
        verdict = "CONTEXTUAL_EDGE"
    elif float(research_metrics.get("exp_lcb", 0.0)) > 0:
        verdict = "WEAK_EDGE"
    else:
        verdict = "NO_EDGE"

    if matrix_df.empty:
        next_bottleneck = "signal_quality"
    elif shadow_rate > 0.50:
        next_bottleneck = "live_feasibility"
    elif wf_pct <= 0.0:
        next_bottleneck = "wf_usability"
    elif mc_pct <= 0.0:
        next_bottleneck = "mc_preconditions"
    elif float(live_metrics.get("exp_lcb", 0.0)) <= 0.0:
        next_bottleneck = "cost_drag_vs_signal"
    else:
        next_bottleneck = "none"

    payload = {
        "stage": "28",
        "run_id": run_id,
        "seed": seed,
        "mode": mode,
        "dry_run": dry_run,
        "head_commit": _git_head(),
        "config_hash": config_hash,
        "data_hash": stable_hash(data_hash_parts, length=16),
        "resolved_end_ts": max(resolved_end) if resolved_end else None,
        "coverage_gate_status": coverage_status,
        "coverage_rows": coverage_rows,
        "requested_symbols": list(symbols),
        "used_symbols": list(used_symbols),
        "timeframes": list(timeframes),
        "window_months": list(window_months),
        "step_months": int(step_months),
        "window_counts": window_counts,
        "funnel_summary": funnel.get("summary", {}),
        "wf_executed_pct": float(wf_pct),
        "mc_trigger_pct": float(mc_pct),
        "qualified_finalists": int(stage_c.shape[0]),
        "top_contextual_edges": top_edges,
        "policy_metrics": {"research": research_metrics, "live": live_metrics},
        "shadow_live_reject_rate": float(shadow_rate),
        "shadow_live_top_reasons": top_reasons,
        "feasibility_summary": {"equity_tiers": [100, 1000, 10000, 100000], "avg_feasible_pct_by_equity": avg_feasible},
        "verdict": verdict,
        "next_bottleneck": next_bottleneck,
        "warnings": warnings,
        "runtime_seconds": float(time.perf_counter() - started),
        **snapshot_meta,
    }
    payload["summary_hash"] = stable_hash(
        {
            "wf": payload["wf_executed_pct"],
            "mc": payload["mc_trigger_pct"],
            "verdict": verdict,
            "data_hash": payload["data_hash"],
            "config_hash": config_hash,
            "snapshot": payload.get("data_snapshot_hash", ""),
            "window_counts": payload["window_counts"],
            "policy_metrics": payload["policy_metrics"],
            "top_contextual_edges": payload["top_contextual_edges"],
        },
        length=16,
    )

    (run_dir / "summary.json").write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    (docs / "stage28_master_summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (docs / "stage28_master_report.md").write_text(_render_master(payload), encoding="utf-8")
    (docs / "stage28_product_spec.md").write_text(_render_product_spec(payload), encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"stage28_dir: {run_dir}")
    print(f"report_md: {docs / 'stage28_master_report.md'}")
    print(f"report_json: {docs / 'stage28_master_summary.json'}")
    print(f"product_spec: {docs / 'stage28_product_spec.md'}")
    print(f"verdict: {payload['verdict']}")


if __name__ == "__main__":
    main()
