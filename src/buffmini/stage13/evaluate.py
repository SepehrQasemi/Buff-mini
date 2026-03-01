"""Stage-13 family engine evaluation runner."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, trend_pullback
from buffmini.config import compute_config_hash
from buffmini.constants import DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.portfolio.monte_carlo import simulate_equity_paths, summarize_mc
from buffmini.signals.composer import compose_signals, normalize_weights
from buffmini.signals.family_base import FamilyContext
from buffmini.signals.registry import build_families, family_names
from buffmini.stage10.evaluate import _build_features
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact
from buffmini.validation.leakage_harness import run_registered_features_harness
from buffmini.validation.walkforward_v2 import build_windows


@dataclass
class FamilyRun:
    symbol: str
    timeframe: str
    family: str
    composer_mode: str
    trade_count: float
    tpm: float
    pf: float
    expectancy: float
    exp_lcb: float
    maxdd: float
    walkforward_executed: bool
    walkforward_usable_windows: int
    walkforward_expected_windows: int
    walkforward_classification: str
    mc_executed: bool
    mc_p_ruin: float
    invalid_reason: str
    classification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "family": self.family,
            "composer_mode": self.composer_mode,
            "trade_count": float(self.trade_count),
            "tpm": float(self.tpm),
            "PF": float(self.pf),
            "expectancy": float(self.expectancy),
            "exp_lcb": float(self.exp_lcb),
            "maxDD": float(self.maxdd),
            "walkforward_executed": bool(self.walkforward_executed),
            "walkforward_usable_windows": int(self.walkforward_usable_windows),
            "walkforward_expected_windows": int(self.walkforward_expected_windows),
            "walkforward_classification": str(self.walkforward_classification),
            "mc_executed": bool(self.mc_executed),
            "mc_p_ruin": float(self.mc_p_ruin) if np.isfinite(self.mc_p_ruin) else None,
            "invalid_reason": str(self.invalid_reason),
            "classification": str(self.classification),
        }


def run_stage13(
    *,
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    families: list[str] | None = None,
    composer_mode: str = "none",
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    stage_tag: str = "13.1",
    report_name: str = "stage13_1_architecture",
    write_docs: bool = True,
    window_months: int | None = None,
    window_offset_months: int = 0,
) -> dict[str, Any]:
    """Run Stage-13 family evaluation and write deterministic artifacts."""

    cfg = json.loads(json.dumps(config))
    stage13_cfg = dict(((cfg.get("evaluation", {}) or {}).get("stage13", {})))
    enabled = bool(stage13_cfg.get("enabled", False))
    resolved_symbols = list(symbols or cfg.get("universe", {}).get("symbols", ["BTC/USDT", "ETH/USDT"]))
    resolved_timeframe = str(timeframe or cfg.get("universe", {}).get("timeframe", "1h"))
    if resolved_timeframe != "1h":
        raise ValueError("Stage-13 currently supports timeframe=1h only")

    features_by_symbol = _build_features(
        config=cfg,
        symbols=resolved_symbols,
        timeframe=resolved_timeframe,
        dry_run=bool(dry_run),
        seed=int(seed),
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    if not features_by_symbol:
        raise RuntimeError("Stage-13: no features loaded")
    if window_months is not None:
        months = max(1, int(window_months))
        offset = max(0, int(window_offset_months))
        trimmed: dict[str, pd.DataFrame] = {}
        for symbol, frame in features_by_symbol.items():
            ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            if ts.dropna().empty:
                continue
            end_ts = ts.max() - pd.Timedelta(days=30 * offset)
            start_ts = end_ts - pd.Timedelta(days=30 * months)
            out = frame.loc[(ts > start_ts) & (ts <= end_ts)].copy().reset_index(drop=True)
            if not out.empty:
                trimmed[symbol] = out
        if trimmed:
            features_by_symbol = trimmed

    data_hash = stable_hash(
        {
            symbol: stable_hash(
                frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="list"),
                length=16,
            )
            for symbol, frame in sorted(features_by_symbol.items())
        },
        length=16,
    )
    config_hash = compute_config_hash(cfg)
    leakage = run_registered_features_harness(rows=520, seed=int(seed), shock_index=420, warmup_max=260, include_futures_extras=True)

    started = time.perf_counter()
    baseline_rows, baseline_hash = _run_stage9_baseline(
        features_by_symbol=features_by_symbol,
        cfg=cfg,
        timeframe=resolved_timeframe,
    )
    runs: list[FamilyRun] = []
    diagnostics_rows: list[dict[str, Any]] = []

    if enabled:
        enabled_families = list(families or stage13_cfg.get("families", {}).get("enabled", family_names()))
        family_map = build_families(enabled=enabled_families, cfg=cfg)
        composer_cfg = dict(stage13_cfg.get("composer", {}))
        mode = str(composer_mode if composer_mode != "none" else composer_cfg.get("mode", "none")).strip().lower()
        weights = normalize_weights(dict(composer_cfg.get("weights", {})), list(family_map.keys()))
        for symbol, frame in sorted(features_by_symbol.items()):
            ctx = FamilyContext(
                symbol=str(symbol),
                timeframe=resolved_timeframe,
                seed=int(seed),
                config=cfg,
                params=stage13_cfg,
            )
            family_outputs: dict[str, pd.DataFrame] = {}
            for family_name, family in family_map.items():
                scores = family.compute_scores(frame, ctx)
                output = family.propose_entries(scores, frame, ctx)
                family_outputs[family_name] = output
                diag = dict(family.diagnostics(frame, ctx))
                diag.update({"symbol": symbol, "timeframe": resolved_timeframe, "family": family_name})
                diagnostics_rows.append(diag)
                run = _evaluate_family_output(
                    frame=frame,
                    signal_output=output,
                    symbol=symbol,
                    family_label=family_name,
                    composer_mode="none",
                    cfg=cfg,
                    seed=int(seed),
                )
                runs.append(run)
            if mode in {"vote", "weighted_sum", "gated"} and len(family_outputs) >= 2:
                comp_output = compose_signals(
                    family_outputs=family_outputs,
                    mode=mode,
                    weights=weights,
                    gated_config=dict(composer_cfg.get("gated", {})),
                )
                runs.append(
                    _evaluate_family_output(
                        frame=frame,
                        signal_output=comp_output,
                        symbol=symbol,
                        family_label="combined",
                        composer_mode=mode,
                        cfg=cfg,
                        seed=int(seed),
                    )
                )
    else:
        for row in baseline_rows:
            runs.append(
                FamilyRun(
                    symbol=str(row["symbol"]),
                    timeframe=resolved_timeframe,
                    family="stage9_baseline",
                    composer_mode="none",
                    trade_count=float(row["trade_count"]),
                    tpm=float(row["tpm"]),
                    pf=float(row["PF"]),
                    expectancy=float(row["expectancy"]),
                    exp_lcb=float(row["exp_lcb"]),
                    maxdd=float(row["maxDD"]),
                    walkforward_executed=False,
                    walkforward_usable_windows=0,
                    walkforward_expected_windows=0,
                    walkforward_classification="DISABLED",
                    mc_executed=False,
                    mc_p_ruin=float("nan"),
                    invalid_reason="STAGE13_DISABLED",
                    classification="INSUFFICIENT_DATA",
                )
            )

    rows_df = pd.DataFrame([item.to_dict() for item in runs])
    gate_metrics = _gate_metrics(rows_df)
    runtime_seconds = float(time.perf_counter() - started)
    top_rows = rows_df.sort_values(["exp_lcb", "PF", "trade_count"], ascending=[False, False, False]).head(10)
    summary = {
        "stage": "13",
        "stage_tag": str(stage_tag),
        "run_id": "",
        "enabled": bool(enabled),
        "symbols": resolved_symbols,
        "timeframe": resolved_timeframe,
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "config_hash": config_hash,
        "data_hash": data_hash,
        "baseline_hash": baseline_hash,
        "leakage": leakage,
        "metrics": gate_metrics,
        "top_rows": top_rows.to_dict(orient="records"),
        "classification": _overall_classification(rows_df, gate_metrics=gate_metrics),
        "runtime_seconds": runtime_seconds,
        "warnings": _summary_warnings(rows_df, gate_metrics),
    }
    run_id = f"{utc_now_compact()}_{stable_hash(summary, length=12)}_stage13"
    summary["run_id"] = run_id

    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_summary = _json_safe(summary)
    rows_df.to_csv(run_dir / "stage13_family_results.csv", index=False)
    pd.DataFrame(diagnostics_rows).to_csv(run_dir / "stage13_family_diagnostics.csv", index=False)
    (run_dir / "stage13_summary.json").write_text(json.dumps(safe_summary, indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2, allow_nan=False), encoding="utf-8")

    report_md = docs_dir / f"{report_name}_report.md"
    report_json = docs_dir / f"{report_name}_summary.json"
    if bool(write_docs):
        docs_dir.mkdir(parents=True, exist_ok=True)
        _write_stage13_report(report_md=report_md, report_json=report_json, summary=safe_summary, rows=rows_df)
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "summary": safe_summary,
        "rows": rows_df,
        "report_md": report_md,
        "report_json": report_json,
    }


def validate_stage13_summary_schema(payload: dict[str, Any]) -> None:
    required = {
        "stage",
        "stage_tag",
        "run_id",
        "enabled",
        "symbols",
        "timeframe",
        "seed",
        "dry_run",
        "config_hash",
        "data_hash",
        "baseline_hash",
        "leakage",
        "metrics",
        "classification",
        "runtime_seconds",
        "warnings",
    }
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise ValueError(f"Missing Stage-13 summary keys: {missing}")
    if str(payload["stage"]) != "13":
        raise ValueError("stage must be '13'")
    if str(payload["classification"]) not in {"ROBUST_EDGE", "WEAK_EDGE", "NO_EDGE", "INSUFFICIENT_DATA"}:
        raise ValueError("classification must be ROBUST_EDGE/WEAK_EDGE/NO_EDGE/INSUFFICIENT_DATA")
    metrics = dict(payload["metrics"])
    for key in (
        "zero_trade_pct",
        "invalid_pct",
        "walkforward_executed_true_pct",
        "mc_trigger_rate",
        "trade_count",
        "tpm",
    ):
        float(metrics.get(key, 0.0))


def _run_stage9_baseline(features_by_symbol: dict[str, pd.DataFrame], cfg: dict[str, Any], timeframe: str) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    strategy = trend_pullback()
    eval_cfg = ((cfg.get("evaluation", {}) or {}).get("stage10", {}).get("evaluation", {}))
    for symbol, frame in sorted(features_by_symbol.items()):
        work = frame.copy()
        work["signal"] = generate_signals(work, strategy=strategy, gating_mode="none")
        result = run_backtest(
            frame=work,
            strategy_name=strategy.name,
            symbol=symbol,
            signal_col="signal",
            max_hold_bars=int(eval_cfg.get("max_hold_bars", 24)),
            stop_atr_multiple=float(eval_cfg.get("stop_atr_multiple", 1.5)),
            take_profit_atr_multiple=float(eval_cfg.get("take_profit_atr_multiple", 3.0)),
            round_trip_cost_pct=float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1)),
            slippage_pct=float(cfg.get("costs", {}).get("slippage_pct", 0.0005)),
            cost_model_cfg=cfg.get("cost_model", {}),
        )
        metrics = _metrics(result=result, frame=work)
        rows.append(
            {
                "symbol": symbol,
                "trade_count": metrics["trade_count"],
                "tpm": metrics["tpm"],
                "PF": metrics["PF"],
                "expectancy": metrics["expectancy"],
                "exp_lcb": metrics["exp_lcb"],
                "maxDD": metrics["maxDD"],
                "trades_hash": stable_hash(result.trades.to_dict(orient="records"), length=16),
                "equity_hash": stable_hash(result.equity_curve.to_dict(orient="records"), length=16),
            }
        )
    baseline_hash = stable_hash(rows, length=16)
    return rows, baseline_hash


def _evaluate_family_output(
    *,
    frame: pd.DataFrame,
    signal_output: pd.DataFrame,
    symbol: str,
    family_label: str,
    composer_mode: str,
    cfg: dict[str, Any],
    seed: int,
) -> FamilyRun:
    work = frame.copy()
    work["signal"] = pd.to_numeric(signal_output.get("signal", 0), errors="coerce").fillna(0).astype(int)
    eval_cfg = ((cfg.get("evaluation", {}) or {}).get("stage10", {}).get("evaluation", {}))
    backtest = run_backtest(
        frame=work,
        strategy_name=f"Stage13::{family_label}",
        symbol=str(symbol),
        signal_col="signal",
        max_hold_bars=int(eval_cfg.get("max_hold_bars", 24)),
        stop_atr_multiple=float(eval_cfg.get("stop_atr_multiple", 1.5)),
        take_profit_atr_multiple=float(eval_cfg.get("take_profit_atr_multiple", 3.0)),
        round_trip_cost_pct=float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1)),
        slippage_pct=float(cfg.get("costs", {}).get("slippage_pct", 0.0005)),
        cost_model_cfg=cfg.get("cost_model", {}),
    )
    met = _metrics(result=backtest, frame=work)
    wf = _walkforward_eval(
        frame=work,
        cfg=cfg,
        seed=int(seed),
    )
    mc = _mc_eval(
        trade_pnls=pd.to_numeric(backtest.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float),
        cfg=cfg,
        seed=int(seed),
    )
    invalid_reason = "VALID"
    if float(met["trade_count"]) <= 0.0:
        invalid_reason = "ZERO_TRADE"
    elif not bool(wf["walkforward_executed"]):
        invalid_reason = "NO_WF"
    elif str(wf["classification"]) == "INSUFFICIENT_DATA":
        invalid_reason = "LOW_USABLE_WINDOWS"
    elif not bool(mc["executed"]):
        invalid_reason = "NO_MC"
    classification = _classify_run(
        trade_count=float(met["trade_count"]),
        pf=float(met["PF"]),
        exp_lcb=float(met["exp_lcb"]),
        maxdd=float(met["maxDD"]),
        wf_classification=str(wf["classification"]),
        mc_executed=bool(mc["executed"]),
        mc_p_ruin=float(mc["p_ruin"]) if mc["executed"] else float("nan"),
    )
    return FamilyRun(
        symbol=str(symbol),
        timeframe=str(frame.attrs.get("timeframe", "1h")),
        family=str(family_label),
        composer_mode=str(composer_mode),
        trade_count=float(met["trade_count"]),
        tpm=float(met["tpm"]),
        pf=float(met["PF"]),
        expectancy=float(met["expectancy"]),
        exp_lcb=float(met["exp_lcb"]),
        maxdd=float(met["maxDD"]),
        walkforward_executed=bool(wf["walkforward_executed"]),
        walkforward_usable_windows=int(wf["usable_windows"]),
        walkforward_expected_windows=int(wf["expected_windows"]),
        walkforward_classification=str(wf["classification"]),
        mc_executed=bool(mc["executed"]),
        mc_p_ruin=float(mc["p_ruin"]) if mc["executed"] else float("nan"),
        invalid_reason=invalid_reason,
        classification=classification,
    )


def _metrics(*, result: Any, frame: pd.DataFrame) -> dict[str, float]:
    trade_count = _finite(result.metrics.get("trade_count", 0.0), default=0.0)
    expectancy = _finite(result.metrics.get("expectancy", 0.0), default=0.0)
    pf = _finite(result.metrics.get("profit_factor", 0.0), default=0.0, clip=10.0)
    maxdd = _finite(result.metrics.get("max_drawdown", 0.0), default=0.0)
    pnl = pd.to_numeric(result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    exp_lcb = _exp_lcb(pnl)
    months = _months(frame)
    tpm = float(trade_count / months) if months > 0 else 0.0
    return {
        "trade_count": float(trade_count),
        "tpm": float(tpm),
        "PF": float(pf),
        "expectancy": float(expectancy),
        "exp_lcb": float(exp_lcb),
        "maxDD": float(maxdd),
    }


def _walkforward_eval(*, frame: pd.DataFrame, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    wf_cfg = (((cfg.get("evaluation", {}) or {}).get("stage8", {}) or {}).get("walkforward_v2", {}))
    eval_cfg = ((cfg.get("evaluation", {}) or {}).get("stage10", {}).get("evaluation", {}))
    if frame.empty:
        return {"walkforward_executed": False, "expected_windows": 0, "usable_windows": 0, "classification": "INSUFFICIENT_DATA"}
    windows = build_windows(
        start_ts=frame["timestamp"].iloc[0],
        end_ts=frame["timestamp"].iloc[-1],
        train_days=int(wf_cfg.get("train_days", 180)),
        holdout_days=int(wf_cfg.get("holdout_days", 30)),
        forward_days=int(wf_cfg.get("forward_days", 30)),
        step_days=int(wf_cfg.get("step_days", 30)),
        reserve_tail_days=int(wf_cfg.get("reserve_tail_days", 0)),
    )
    if not windows:
        return {"walkforward_executed": False, "expected_windows": 0, "usable_windows": 0, "classification": "INSUFFICIENT_DATA"}
    min_trades = int(max(1, wf_cfg.get("min_trades", 10)))
    min_exposure = float(max(0.0, wf_cfg.get("min_exposure", 0.01)))
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    usable = 0
    executed = 0
    for w in windows:
        forward = frame.loc[(ts >= w.forward_start) & (ts < w.forward_end)].copy().reset_index(drop=True)
        if forward.empty:
            continue
        executed += 1
        result = run_backtest(
            frame=forward,
            strategy_name="Stage13::WF",
            symbol=str(frame.attrs.get("symbol", "UNKNOWN")),
            signal_col="signal",
            max_hold_bars=int(eval_cfg.get("max_hold_bars", 24)),
            stop_atr_multiple=float(eval_cfg.get("stop_atr_multiple", 1.5)),
            take_profit_atr_multiple=float(eval_cfg.get("take_profit_atr_multiple", 3.0)),
            round_trip_cost_pct=float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1)),
            slippage_pct=float(cfg.get("costs", {}).get("slippage_pct", 0.0005)),
            cost_model_cfg=cfg.get("cost_model", {}),
        )
        trades = result.trades.copy()
        exposure = 0.0
        if not trades.empty and "bars_held" in trades.columns:
            exposure = float(pd.to_numeric(trades["bars_held"], errors="coerce").fillna(0.0).sum() / max(1, len(forward)))
        if float(result.metrics.get("trade_count", 0.0)) >= float(min_trades) and exposure >= min_exposure:
            usable += 1
    classification = "INSUFFICIENT_DATA"
    if executed > 0:
        usable_ratio = float(usable / executed)
        if usable_ratio >= 0.6:
            classification = "STABLE"
        elif usable_ratio > 0.0:
            classification = "UNSTABLE"
        else:
            classification = "INSUFFICIENT_DATA"
    return {
        "walkforward_executed": bool(executed > 0),
        "expected_windows": int(len(windows)),
        "usable_windows": int(usable),
        "classification": str(classification),
    }


def _mc_eval(*, trade_pnls: np.ndarray, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    arr = np.asarray(trade_pnls, dtype=float)
    if arr.size <= 0:
        return {"executed": False, "p_ruin": math.nan}
    mc_cfg = (((cfg.get("evaluation", {}) or {}).get("stage12", {}) or {}).get("monte_carlo", {}))
    if not bool(mc_cfg.get("enabled", True)):
        return {"executed": False, "p_ruin": math.nan}
    paths = simulate_equity_paths(
        trade_pnls=arr,
        n_paths=int(mc_cfg.get("n_paths", 1000)),
        method=str(mc_cfg.get("bootstrap", "block")),
        seed=int(seed),
        initial_equity=float(mc_cfg.get("initial_equity", 10000.0)),
        leverage=1.0,
        block_size_trades=int(mc_cfg.get("block_size_trades", 10)),
    )
    summary = summarize_mc(
        paths_results=paths,
        initial_equity=float(mc_cfg.get("initial_equity", 10000.0)),
        ruin_dd_threshold=float(mc_cfg.get("ruin_dd_threshold", 0.5)),
    )
    return {
        "executed": True,
        "p_ruin": float(summary["tail_probabilities"]["p_ruin"]),
    }


def _classify_run(
    *,
    trade_count: float,
    pf: float,
    exp_lcb: float,
    maxdd: float,
    wf_classification: str,
    mc_executed: bool,
    mc_p_ruin: float,
) -> str:
    if float(trade_count) <= 0.0:
        return "INSUFFICIENT_DATA"
    if str(wf_classification) == "INSUFFICIENT_DATA":
        return "INSUFFICIENT_DATA"
    if float(exp_lcb) > 0.0 and float(pf) > 1.1 and float(maxdd) < 0.25 and str(wf_classification) == "STABLE":
        if not mc_executed or not np.isfinite(mc_p_ruin) or float(mc_p_ruin) <= 0.05:
            return "ROBUST_EDGE"
        return "WEAK_EDGE"
    if float(exp_lcb) > 0.0 and float(pf) > 1.0:
        return "WEAK_EDGE"
    return "NO_EDGE"


def _gate_metrics(rows_df: pd.DataFrame) -> dict[str, float]:
    if rows_df.empty:
        return {
            "zero_trade_pct": 100.0,
            "invalid_pct": 100.0,
            "walkforward_executed_true_pct": 0.0,
            "mc_trigger_rate": 0.0,
            "tpm": 0.0,
            "trade_count": 0.0,
        }
    trade_count = pd.to_numeric(rows_df["trade_count"], errors="coerce").fillna(0.0)
    invalid = rows_df["classification"].astype(str).isin({"NO_EDGE", "INSUFFICIENT_DATA"})
    wf_ex = rows_df["walkforward_executed"].astype(bool)
    mc = rows_df["mc_executed"].astype(bool)
    tpm = pd.to_numeric(rows_df["tpm"], errors="coerce").fillna(0.0)
    return {
        "zero_trade_pct": float((trade_count <= 0.0).mean() * 100.0),
        "invalid_pct": float(invalid.mean() * 100.0),
        "walkforward_executed_true_pct": float(wf_ex.mean() * 100.0),
        "mc_trigger_rate": float(mc.mean() * 100.0),
        "tpm": float(tpm.mean()),
        "trade_count": float(trade_count.mean()),
    }


def _overall_classification(rows_df: pd.DataFrame, gate_metrics: dict[str, float]) -> str:
    if rows_df.empty:
        return "INSUFFICIENT_DATA"
    any_robust = bool((rows_df["classification"] == "ROBUST_EDGE").any())
    any_weak = bool((rows_df["classification"] == "WEAK_EDGE").any())
    if any_robust:
        return "ROBUST_EDGE"
    if any_weak:
        return "WEAK_EDGE"
    if float(gate_metrics["walkforward_executed_true_pct"]) <= 0.0 and float(gate_metrics["mc_trigger_rate"]) <= 0.0:
        return "INSUFFICIENT_DATA"
    return "NO_EDGE"


def _summary_warnings(rows_df: pd.DataFrame, gate_metrics: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    if float(gate_metrics["invalid_pct"]) >= 100.0:
        warnings.append("all_configs_invalid")
    if float(gate_metrics["walkforward_executed_true_pct"]) <= 0.0:
        warnings.append("walkforward_not_executed")
    if float(gate_metrics["mc_trigger_rate"]) <= 0.0:
        warnings.append("mc_not_triggered")
    if rows_df.empty:
        warnings.append("no_rows")
    return warnings


def _write_stage13_report(*, report_md: Path, report_json: Path, summary: dict[str, Any], rows: pd.DataFrame) -> None:
    report_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    metrics = dict(summary["metrics"])
    lines: list[str] = []
    lines.append(f"# Stage-{summary['stage_tag']} Report")
    lines.append("")
    lines.append("## 1) What changed")
    lines.append("- Stage-13 family engine contract and evaluation layer executed.")
    lines.append("- Family runs are deterministic and artifact-driven.")
    lines.append("")
    lines.append("## 2) How to run (dry-run + real)")
    lines.append("- dry-run: `python scripts/run_stage13.py --dry-run --seed 42`")
    lines.append("- real: `python scripts/run_stage13.py --seed 42`")
    lines.append("")
    lines.append("## 3) Validation gates & results")
    lines.append(f"- zero_trade_pct: `{float(metrics.get('zero_trade_pct', 0.0)):.6f}`")
    lines.append(f"- invalid_pct: `{float(metrics.get('invalid_pct', 0.0)):.6f}`")
    lines.append(f"- walkforward_executed_true_pct: `{float(metrics.get('walkforward_executed_true_pct', 0.0)):.6f}`")
    lines.append(f"- mc_trigger_rate: `{float(metrics.get('mc_trigger_rate', 0.0)):.6f}`")
    lines.append("")
    lines.append("## 4) Key metrics tables (trade_count, tpm, PF, expectancy, exp_lcb, maxDD, wf, mc)")
    if rows.empty:
        lines.append("- no candidate rows")
    else:
        lines.append("| symbol | family | composer | trade_count | tpm | PF | expectancy | exp_lcb | maxDD | wf_class | mc | class |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")
        for row in rows.sort_values(["exp_lcb", "PF"], ascending=[False, False]).head(25).to_dict(orient="records"):
            lines.append(
                f"| {row.get('symbol','')} | {row.get('family','')} | {row.get('composer_mode','')} | "
                f"{float(row.get('trade_count',0.0)):.2f} | {float(row.get('tpm',0.0)):.2f} | "
                f"{float(row.get('PF',0.0)):.4f} | {float(row.get('expectancy',0.0)):.6f} | "
                f"{float(row.get('exp_lcb',0.0)):.6f} | {float(row.get('maxDD',0.0)):.6f} | "
                f"{row.get('walkforward_classification','')} | {bool(row.get('mc_executed',False))} | {row.get('classification','')} |"
            )
    lines.append("")
    lines.append("## 5) Failures + reasons")
    if summary["warnings"]:
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## 6) Next actions")
    lines.append("- Continue to next Stage-13/14 sub-stage refinement.")
    lines.append("")
    lines.append("## Repro")
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- seed: `{summary['seed']}`")
    lines.append(f"- config_hash: `{summary['config_hash']}`")
    lines.append(f"- data_hash: `{summary['data_hash']}`")
    lines.append(f"- classification: `{summary['classification']}`")
    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _exp_lcb(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean
    std = float(np.std(arr, ddof=0))
    return float(mean - std / math.sqrt(float(arr.size)))


def _months(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    days = float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0)
    return max(days / 30.0, 1e-6)


def _finite(value: Any, default: float = 0.0, clip: float | None = None) -> float:
    try:
        num = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(num):
        return float(default)
    if clip is not None:
        num = min(num, float(clip))
    return float(num)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        num = float(value)
        return num if np.isfinite(num) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def run_stage13_family_sweep(
    *,
    config: dict[str, Any],
    family: str,
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    stage_tag: str = "13.2",
    report_name: str = "stage13_2_price_family",
) -> dict[str, Any]:
    """Run bounded grid sweep for one Stage-13 family."""

    cfg = json.loads(json.dumps(config))
    stage13 = dict(((cfg.get("evaluation", {}) or {}).get("stage13", {})))
    family_key = str(family)
    if family_key not in {"price", "volatility", "flow"}:
        raise ValueError("family must be price|volatility|flow")
    family_cfg = dict(stage13.get(family_key, {}))
    sweep_grid = dict(family_cfg.get("sweep_grid", {}))
    keys = sorted(sweep_grid.keys())
    values = [list(sweep_grid[key]) for key in keys]
    combos: list[dict[str, Any]] = []
    if keys and all(values):
        for row in product(*values):
            combos.append({key: row[idx] for idx, key in enumerate(keys)})
    else:
        combos.append({})
    if len(combos) > 80:
        combos = combos[:80]

    baseline_cfg = json.loads(json.dumps(cfg))
    baseline_cfg.setdefault("evaluation", {}).setdefault("stage13", {})["enabled"] = False
    baseline = run_stage13(
        config=baseline_cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=symbols,
        timeframe=timeframe,
        families=[family_key],
        composer_mode="none",
        runs_root=runs_root,
        docs_dir=docs_dir,
        data_dir=data_dir,
        derived_dir=derived_dir,
        stage_tag=stage_tag,
        report_name=f"{report_name}_baseline",
        write_docs=False,
    )
    baseline_trade_count = float(dict(baseline["summary"].get("metrics", {})).get("trade_count", 0.0))

    sweep_rows: list[dict[str, Any]] = []
    run_ids: list[str] = []
    for idx, combo in enumerate(combos):
        cfg_i = json.loads(json.dumps(cfg))
        cfg_i.setdefault("evaluation", {}).setdefault("stage13", {})["enabled"] = True
        cfg_i["evaluation"]["stage13"].setdefault("families", {})["enabled"] = [family_key]
        cfg_i["evaluation"]["stage13"][family_key] = {
            **dict(cfg_i["evaluation"]["stage13"].get(family_key, {})),
            **combo,
        }
        result = run_stage13(
            config=cfg_i,
            seed=int(seed),
            dry_run=bool(dry_run),
            symbols=symbols,
            timeframe=timeframe,
            families=[family_key],
            composer_mode="none",
            runs_root=runs_root,
            docs_dir=docs_dir,
            data_dir=data_dir,
            derived_dir=derived_dir,
            stage_tag=stage_tag,
            report_name=f"{report_name}_combo_{idx}",
            write_docs=False,
        )
        summary = dict(result["summary"])
        metrics = dict(summary.get("metrics", {}))
        run_ids.append(str(summary["run_id"]))
        rows = result["rows"]
        best = rows.sort_values(["exp_lcb", "PF", "trade_count"], ascending=[False, False, False]).head(1)
        top = best.iloc[0].to_dict() if not best.empty else {}
        sweep_rows.append(
            {
                "combo_idx": idx,
                "run_id": summary["run_id"],
                "params": combo,
                "zero_trade_pct": float(metrics.get("zero_trade_pct", 100.0)),
                "invalid_pct": float(metrics.get("invalid_pct", 100.0)),
                "walkforward_executed_true_pct": float(metrics.get("walkforward_executed_true_pct", 0.0)),
                "mc_trigger_rate": float(metrics.get("mc_trigger_rate", 0.0)),
                "trade_count": float(metrics.get("trade_count", 0.0)),
                "tpm": float(metrics.get("tpm", 0.0)),
                "best_PF": float(top.get("PF", 0.0)) if top else 0.0,
                "best_expectancy": float(top.get("expectancy", 0.0)) if top else 0.0,
                "best_exp_lcb": float(top.get("exp_lcb", 0.0)) if top else 0.0,
                "best_maxDD": float(top.get("maxDD", 0.0)) if top else 0.0,
                "classification": str(summary.get("classification", "NO_EDGE")),
            }
        )

    sweep_df = pd.DataFrame(sweep_rows)
    if sweep_df.empty:
        raise RuntimeError("Stage-13 sweep produced no rows")
    best_row = sweep_df.sort_values(["best_exp_lcb", "best_PF", "trade_count"], ascending=[False, False, False]).iloc[0].to_dict()
    zero_trade_ok = bool((sweep_df["zero_trade_pct"] < 40.0).any())
    trade_count_ratio = float(best_row.get("trade_count", 0.0) / baseline_trade_count) if baseline_trade_count > 0 else 0.0
    classification = str(best_row.get("classification", "NO_EDGE"))
    if family_key == "price":
        min_ratio = float(stage13.get("gates", {}).get("min_trade_count_ratio_vs_baseline", 0.60))
    elif family_key == "volatility":
        min_ratio = 0.40
    else:
        min_ratio = 0.0
    wf_ok = bool((sweep_df["walkforward_executed_true_pct"] > 0.0).any())
    mc_ok = bool((sweep_df["mc_trigger_rate"] > 0.0).any())
    if not zero_trade_ok:
        classification = "NO_EDGE"
    if trade_count_ratio < float(min_ratio):
        classification = "NO_EDGE"
    if family_key == "volatility" and (not wf_ok or not mc_ok):
        classification = "NO_EDGE"

    payload = {
        "git_commit": "",
        "stage": str(stage_tag),
        "family": family_key,
        "run_ids": run_ids,
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "config_hash": compute_config_hash(cfg),
        "data_hash": stable_hash(sweep_df["run_id"].tolist(), length=16),
        "baseline_trade_count": baseline_trade_count,
        "best_combo": _json_safe(best_row),
        "trade_count_ratio_vs_baseline": float(trade_count_ratio),
        "zero_trade_pct_min": float(sweep_df["zero_trade_pct"].min()),
        "walkforward_executed_true_pct_max": float(sweep_df["walkforward_executed_true_pct"].max()),
        "mc_trigger_rate_max": float(sweep_df["mc_trigger_rate"].max()),
        "classification": classification,
        "runtime_seconds": 0.0,
        "warnings": [],
    }
    if not zero_trade_ok:
        payload["warnings"].append("zero_trade_pct_gate_failed")
    if trade_count_ratio < float(min_ratio):
        payload["warnings"].append("trade_count_ratio_gate_failed")
    if family_key == "volatility" and not wf_ok:
        payload["warnings"].append("walkforward_gate_failed")
    if family_key == "volatility" and not mc_ok:
        payload["warnings"].append("mc_gate_failed")

    docs_dir.mkdir(parents=True, exist_ok=True)
    report_json = docs_dir / f"{report_name}_summary.json"
    report_md = docs_dir / f"{report_name}_report.md"
    sweep_table_path = docs_dir / f"{report_name}_table.csv"
    sweep_df.to_csv(sweep_table_path, index=False)
    report_json.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        f"# Stage-{stage_tag} {family_key.title()} Family Report",
        "",
        "## 1) What changed",
        f"- Ran bounded sweep for `{family_key}` family with deterministic configs.",
        "",
        "## 2) How to run (dry-run + real)",
        f"- dry-run: `python scripts/run_stage13.py --substage {stage_tag} --family {family_key} --dry-run --seed 42`",
        f"- real: `python scripts/run_stage13.py --substage {stage_tag} --family {family_key} --seed 42`",
        "",
        "## 3) Validation gates & results",
        f"- zero_trade_pct_min: `{payload['zero_trade_pct_min']:.6f}`",
        f"- trade_count_ratio_vs_baseline: `{payload['trade_count_ratio_vs_baseline']:.6f}`",
        f"- walkforward_executed_true_pct_max: `{payload['walkforward_executed_true_pct_max']:.6f}`",
        f"- mc_trigger_rate_max: `{payload['mc_trigger_rate_max']:.6f}`",
        "",
        "## 4) Key metrics tables (trade_count, tpm, PF, expectancy, exp_lcb, maxDD, wf, mc)",
        f"- sweep table: `{sweep_table_path.as_posix()}`",
        "",
        "## 5) Failures + reasons",
    ]
    if payload["warnings"]:
        for warning in payload["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## 6) Next actions",
            "- Use best family configs in combined composer matrix (Stage-13.5).",
            "",
            "## Summary",
            f"- classification: `{payload['classification']}`",
            f"- best_exp_lcb: `{float(best_row.get('best_exp_lcb', 0.0)):.6f}`",
            f"- best_trade_count: `{float(best_row.get('trade_count', 0.0)):.2f}`",
        ]
    )
    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return {
        "summary": payload,
        "table": sweep_df,
        "report_md": report_md,
        "report_json": report_json,
    }


def run_stage13_combined_matrix(
    *,
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    stage_tag: str = "13.5",
    report_name: str = "stage13_5_combined",
) -> dict[str, Any]:
    """Run family-alone/pairwise/all matrix with composer modes."""

    cfg = json.loads(json.dumps(config))
    family_sets = [
        ["price"],
        ["volatility"],
        ["flow"],
        ["price", "volatility"],
        ["price", "flow"],
        ["volatility", "flow"],
        ["price", "volatility", "flow"],
    ]
    rows: list[dict[str, Any]] = []
    run_ids: list[str] = []
    for fams in family_sets:
        modes = ["none"] if len(fams) == 1 else ["vote", "weighted_sum", "gated"]
        for mode in modes:
            cfg_i = json.loads(json.dumps(cfg))
            cfg_i.setdefault("evaluation", {}).setdefault("stage13", {})["enabled"] = True
            cfg_i["evaluation"]["stage13"].setdefault("families", {})["enabled"] = list(fams)
            result = run_stage13(
                config=cfg_i,
                seed=int(seed),
                dry_run=bool(dry_run),
                symbols=symbols,
                timeframe=timeframe,
                families=list(fams),
                composer_mode=mode,
                runs_root=runs_root,
                docs_dir=docs_dir,
                data_dir=data_dir,
                derived_dir=derived_dir,
                stage_tag=stage_tag,
                report_name=f"{report_name}_{'_'.join(fams)}_{mode}",
                write_docs=False,
            )
            summary = dict(result["summary"])
            metrics = dict(summary.get("metrics", {}))
            run_ids.append(str(summary["run_id"]))
            rows.append(
                {
                    "families": ",".join(fams),
                    "composer_mode": mode,
                    "run_id": summary["run_id"],
                    "classification": summary["classification"],
                    "zero_trade_pct": float(metrics.get("zero_trade_pct", 100.0)),
                    "invalid_pct": float(metrics.get("invalid_pct", 100.0)),
                    "walkforward_executed_true_pct": float(metrics.get("walkforward_executed_true_pct", 0.0)),
                    "mc_trigger_rate": float(metrics.get("mc_trigger_rate", 0.0)),
                    "trade_count": float(metrics.get("trade_count", 0.0)),
                    "tpm": float(metrics.get("tpm", 0.0)),
                    "best_exp_lcb": float(result["rows"]["exp_lcb"].max()) if not result["rows"].empty else 0.0,
                    "best_pf": float(result["rows"]["PF"].max()) if not result["rows"].empty else 0.0,
                }
            )

    matrix_df = pd.DataFrame(rows)
    pass_gate = bool(
        (
            (matrix_df["walkforward_executed_true_pct"] >= 50.0)
            & (matrix_df["mc_trigger_rate"] > 0.0)
            & (matrix_df["invalid_pct"] < 50.0)
        ).any()
    )
    classification = "ROBUST_EDGE" if pass_gate else "NO_EDGE"
    best_row = matrix_df.sort_values(["best_exp_lcb", "best_pf", "trade_count"], ascending=[False, False, False]).iloc[0].to_dict()

    interaction = _interaction_analysis(
        config=cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=symbols,
        timeframe=timeframe,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    overlap_df = pd.DataFrame(interaction["overlap_rows"])
    corr_df = pd.DataFrame(interaction["correlation_rows"])
    conflict_df = pd.DataFrame(interaction["conflict_rows"])

    payload = {
        "git_commit": "",
        "stage": str(stage_tag),
        "run_ids": run_ids,
        "config_hash": compute_config_hash(cfg),
        "data_hash": stable_hash(run_ids, length=16),
        "classification": classification,
        "gate_pass": pass_gate,
        "best": _json_safe(best_row),
        "warnings": [] if pass_gate else ["combined_gate_failed"],
    }
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_json = docs_dir / f"{report_name}_summary.json"
    report_md = docs_dir / f"{report_name}_report.md"
    matrix_path = docs_dir / f"{report_name}_matrix.csv"
    overlap_path = docs_dir / f"{report_name}_overlap.csv"
    corr_path = docs_dir / f"{report_name}_return_corr.csv"
    conflict_path = docs_dir / f"{report_name}_conflict.csv"
    matrix_df.to_csv(matrix_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)
    corr_df.to_csv(corr_path, index=False)
    conflict_df.to_csv(conflict_path, index=False)
    report_json.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        f"# Stage-{stage_tag} Combined Composer Report",
        "",
        "## 1) What changed",
        "- Ran family-alone, pairwise, and all-family composer matrix.",
        "- Added interaction analysis: overlap, return-correlation, conflict rate.",
        "",
        "## 2) How to run (dry-run + real)",
        f"- dry-run: `python scripts/run_stage13.py --substage {stage_tag} --dry-run --seed 42`",
        f"- real: `python scripts/run_stage13.py --substage {stage_tag} --seed 42`",
        "",
        "## 3) Validation gates & results",
        f"- gate_pass: `{pass_gate}`",
        f"- classification: `{classification}`",
        "",
        "## 4) Key metrics tables (trade_count, tpm, PF, expectancy, exp_lcb, maxDD, wf, mc)",
        f"- matrix: `{matrix_path.as_posix()}`",
        f"- overlap: `{overlap_path.as_posix()}`",
        f"- return correlation: `{corr_path.as_posix()}`",
        f"- conflict: `{conflict_path.as_posix()}`",
        "",
        "## 5) Failures + reasons",
    ]
    if payload["warnings"]:
        lines.extend([f"- {item}" for item in payload["warnings"]])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## 6) Next actions",
            "- Feed best combined mode into robustness sweeps (Stage-13.6).",
            "",
            "## Summary",
            f"- best families: `{best_row.get('families','')}`",
            f"- best composer_mode: `{best_row.get('composer_mode','')}`",
            f"- best_exp_lcb: `{float(best_row.get('best_exp_lcb', 0.0)):.6f}`",
        ]
    )
    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return {
        "summary": payload,
        "table": matrix_df,
        "report_md": report_md,
        "report_json": report_json,
    }


def _interaction_analysis(
    *,
    config: dict[str, Any],
    seed: int,
    dry_run: bool,
    symbols: list[str] | None,
    timeframe: str,
    data_dir: Path,
    derived_dir: Path,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    resolved_symbols = list(symbols or cfg.get("universe", {}).get("symbols", ["BTC/USDT", "ETH/USDT"]))
    features_by_symbol = _build_features(
        config=cfg,
        symbols=resolved_symbols,
        timeframe=str(timeframe),
        dry_run=bool(dry_run),
        seed=int(seed),
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    overlap_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    for symbol, frame in sorted(features_by_symbol.items()):
        fams = build_families(enabled=["price", "volatility", "flow"], cfg=cfg)
        outputs: dict[str, pd.DataFrame] = {}
        ctx = FamilyContext(symbol=str(symbol), timeframe=str(timeframe), seed=int(seed), config=cfg, params={})
        for name, family in fams.items():
            out = family.propose_entries(family.compute_scores(frame, ctx), frame, ctx)
            outputs[name] = out
        fam_names = sorted(outputs.keys())
        returns = pd.to_numeric(frame["close"], errors="coerce").pct_change().fillna(0.0)
        for i, a in enumerate(fam_names):
            for b in fam_names[i + 1 :]:
                sig_a = pd.to_numeric(outputs[a]["direction"], errors="coerce").fillna(0.0)
                sig_b = pd.to_numeric(outputs[b]["direction"], errors="coerce").fillna(0.0)
                act_a = sig_a != 0
                act_b = sig_b != 0
                both = act_a & act_b
                union = act_a | act_b
                overlap = float(both.sum() / max(1, union.sum()))
                conflict = float(((sig_a * sig_b) < 0).sum() / max(1, both.sum()))
                ret_a = (sig_a.shift(1).fillna(0.0) * returns).astype(float)
                ret_b = (sig_b.shift(1).fillna(0.0) * returns).astype(float)
                corr = float(ret_a.corr(ret_b)) if ret_a.std(ddof=0) > 0 and ret_b.std(ddof=0) > 0 else 0.0
                overlap_rows.append({"symbol": symbol, "pair": f"{a}|{b}", "overlap": overlap})
                corr_rows.append({"symbol": symbol, "pair": f"{a}|{b}", "return_corr": corr})
                conflict_rows.append({"symbol": symbol, "pair": f"{a}|{b}", "conflict_rate": conflict})
    return {
        "overlap_rows": overlap_rows,
        "correlation_rows": corr_rows,
        "conflict_rows": conflict_rows,
    }


def run_stage13_robustness(
    *,
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    stage_tag: str = "13.6",
    report_name: str = "stage13_6_robustness",
) -> dict[str, Any]:
    """Run deterministic robustness sweeps (cost/drag/regime-stress proxy)."""

    cfg = json.loads(json.dumps(config))
    matrix = run_stage13_combined_matrix(
        config=cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=symbols,
        timeframe=timeframe,
        runs_root=runs_root,
        docs_dir=docs_dir,
        data_dir=data_dir,
        derived_dir=derived_dir,
        stage_tag="13.5",
        report_name="stage13_5_combined",
    )["table"]
    if matrix.empty:
        raise RuntimeError("Stage-13.6 requires Stage-13.5 matrix rows")

    top = matrix.sort_values(["best_exp_lcb", "best_pf", "trade_count"], ascending=[False, False, False]).head(5)
    cost_scenarios = (((cfg.get("evaluation", {}) or {}).get("stage12", {}) or {}).get("cost_scenarios", {}))
    inst_penalties = {"ROBUST_EDGE": 0.0, "WEAK_EDGE": 0.5, "NO_EDGE": 1.0, "INSUFFICIENT_DATA": 1.0}
    rows: list[dict[str, Any]] = []
    for idx, base in top.reset_index(drop=True).iterrows():
        families = [item for item in str(base["families"]).split(",") if item]
        composer_mode = str(base["composer_mode"])
        if composer_mode == "none":
            composer_mode = "weighted_sum"
        exp_lcb_by_cost: list[float] = []
        for cost_level in ("low", "realistic", "high"):
            cfg_i = json.loads(json.dumps(cfg))
            cfg_i.setdefault("evaluation", {}).setdefault("stage13", {})["enabled"] = True
            cfg_i["evaluation"]["stage13"].setdefault("families", {})["enabled"] = families
            if cost_level != "realistic":
                override = dict(cost_scenarios.get(cost_level, {}))
                if isinstance(override, dict):
                    v2 = cfg_i.setdefault("cost_model", {}).setdefault("v2", {})
                    v2["slippage_bps_base"] = float(override.get("slippage_bps_base", v2.get("slippage_bps_base", 0.5)))
                    v2["slippage_bps_vol_mult"] = float(override.get("slippage_bps_vol_mult", v2.get("slippage_bps_vol_mult", 2.0)))
                    v2["spread_bps"] = float(override.get("spread_bps", v2.get("spread_bps", 0.5)))
                    v2["delay_bars"] = int(override.get("delay_bars", v2.get("delay_bars", 0)))
            result = run_stage13(
                config=cfg_i,
                seed=int(seed) + idx,
                dry_run=bool(dry_run),
                symbols=symbols,
                timeframe=timeframe,
                families=families,
                composer_mode=composer_mode,
                runs_root=runs_root,
                docs_dir=docs_dir,
                data_dir=data_dir,
                derived_dir=derived_dir,
                stage_tag=stage_tag,
                report_name=f"{report_name}_tmp_{idx}_{cost_level}",
                write_docs=False,
            )
            best_exp = float(result["rows"]["exp_lcb"].max()) if not result["rows"].empty else 0.0
            exp_lcb_by_cost.append(best_exp)
            if cost_level == "realistic":
                base_exp = best_exp
            rows.append(
                {
                    "candidate_idx": int(idx),
                    "families": ",".join(families),
                    "composer_mode": composer_mode,
                    "cost_level": cost_level,
                    "best_exp_lcb": best_exp,
                    "classification": str(result["summary"]["classification"]),
                    "trade_count": float(result["summary"]["metrics"]["trade_count"]),
                    "walkforward_executed_true_pct": float(result["summary"]["metrics"]["walkforward_executed_true_pct"]),
                }
            )
        # Drag stress run: +delay and +spread/slippage
        cfg_drag = json.loads(json.dumps(cfg))
        v2 = cfg_drag.setdefault("cost_model", {}).setdefault("v2", {})
        v2["delay_bars"] = int(v2.get("delay_bars", 0)) + 1
        v2["spread_bps"] = float(v2.get("spread_bps", 0.5)) + 1.0
        v2["slippage_bps_base"] = float(v2.get("slippage_bps_base", 0.5)) + 1.0
        drag_run = run_stage13(
            config=cfg_drag,
            seed=int(seed) + 10_000 + idx,
            dry_run=bool(dry_run),
            symbols=symbols,
            timeframe=timeframe,
            families=families,
            composer_mode=composer_mode,
            runs_root=runs_root,
            docs_dir=docs_dir,
            data_dir=data_dir,
            derived_dir=derived_dir,
            stage_tag=stage_tag,
            report_name=f"{report_name}_tmp_{idx}_drag",
            write_docs=False,
        )
        drag_exp = float(drag_run["rows"]["exp_lcb"].max()) if not drag_run["rows"].empty else 0.0
        drag_penalty = max(0.0, float(base_exp - drag_exp))
        regime_stress_penalty = float(np.std(np.asarray(exp_lcb_by_cost, dtype=float))) if exp_lcb_by_cost else 0.0
        inst_penalty = inst_penalties.get(str(base.get("classification", "NO_EDGE")), 1.0)
        robust_score = float(np.median(np.asarray(exp_lcb_by_cost, dtype=float)) - drag_penalty - inst_penalty - regime_stress_penalty)
        rows.append(
            {
                "candidate_idx": int(idx),
                "families": ",".join(families),
                "composer_mode": composer_mode,
                "cost_level": "robust_summary",
                "best_exp_lcb": float(np.median(np.asarray(exp_lcb_by_cost, dtype=float))) if exp_lcb_by_cost else 0.0,
                "classification": "ROBUST_EDGE" if robust_score > 0.0 else "NO_EDGE",
                "trade_count": float(base.get("trade_count", 0.0)),
                "walkforward_executed_true_pct": float(base.get("walkforward_executed_true_pct", 0.0)),
                "drag_penalty": drag_penalty,
                "instability_penalty": inst_penalty,
                "regime_shift_penalty": regime_stress_penalty,
                "robust_score": robust_score,
            }
        )
    table = pd.DataFrame(rows)
    robust_table = table.loc[table["cost_level"] == "robust_summary"].copy()
    if robust_table.empty:
        classification = "NO_EDGE"
        best = {}
    else:
        best_row = robust_table.sort_values("robust_score", ascending=False).iloc[0].to_dict()
        classification = "ROBUST_EDGE" if float(best_row.get("robust_score", 0.0)) > 0 else "NO_EDGE"
        if int((robust_table["robust_score"] > 0).sum()) == 1:
            classification = "WEAK_EDGE"
        best = best_row

    payload = {
        "git_commit": "",
        "stage": str(stage_tag),
        "classification": classification,
        "best": _json_safe(best),
        "warnings": [] if classification != "NO_EDGE" else ["cost_or_drag_collapse"],
    }
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_json = docs_dir / f"{report_name}_summary.json"
    report_md = docs_dir / f"{report_name}_report.md"
    table_path = docs_dir / f"{report_name}_table.csv"
    table.to_csv(table_path, index=False)
    report_json.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        f"# Stage-{stage_tag} Robustness Report",
        "",
        "## 1) What changed",
        "- Added cost sensitivity, execution drag, and deterministic regime-stress proxy sweeps.",
        "",
        "## 2) How to run (dry-run + real)",
        f"- dry-run: `python scripts/run_stage13.py --substage {stage_tag} --dry-run --seed 42`",
        f"- real: `python scripts/run_stage13.py --substage {stage_tag} --seed 42`",
        "",
        "## 3) Validation gates & results",
        f"- classification: `{classification}`",
        f"- best robust_score: `{float(best.get('robust_score', 0.0)):.6f}`",
        "",
        "## 4) Key metrics tables (trade_count, tpm, PF, expectancy, exp_lcb, maxDD, wf, mc)",
        f"- table: `{table_path.as_posix()}`",
        "",
        "## 5) Failures + reasons",
    ]
    if payload["warnings"]:
        lines.extend([f"- {warning}" for warning in payload["warnings"]])
    else:
        lines.append("- none")
    lines.extend(["", "## 6) Next actions", "- Use best robust config for multi-horizon validation."])
    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return {
        "summary": payload,
        "table": table,
        "report_md": report_md,
        "report_json": report_json,
    }


def run_stage13_multihorizon(
    *,
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    stage_tag: str = "13.7",
    report_name: str = "stage13_7_multihorizon",
) -> dict[str, Any]:
    """Run rolling multi-horizon validation (3m/6m/12m)."""

    cfg = json.loads(json.dumps(config))
    robust = run_stage13_robustness(
        config=cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=symbols,
        timeframe=timeframe,
        runs_root=runs_root,
        docs_dir=docs_dir,
        data_dir=data_dir,
        derived_dir=derived_dir,
        stage_tag="13.6",
        report_name="stage13_6_robustness",
    )
    best = dict(robust["summary"].get("best", {}))
    families = [item for item in str(best.get("families", "price,volatility")).split(",") if item] or ["price", "volatility"]
    composer_mode = str(best.get("composer_mode", "weighted_sum"))
    horizons = [3, 6, 12]
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        for offset in range(0, horizon * 3, horizon):
            result = run_stage13(
                config=cfg,
                seed=int(seed) + horizon + offset,
                dry_run=bool(dry_run),
                symbols=symbols,
                timeframe=timeframe,
                families=families,
                composer_mode=composer_mode,
                runs_root=runs_root,
                docs_dir=docs_dir,
                data_dir=data_dir,
                derived_dir=derived_dir,
                stage_tag=stage_tag,
                report_name=f"{report_name}_{horizon}m_{offset}",
                write_docs=False,
                window_months=horizon,
                window_offset_months=offset,
            )
            metrics = dict(result["summary"]["metrics"])
            rows.append(
                {
                    "horizon_months": horizon,
                    "offset_months": offset,
                    "run_id": result["summary"]["run_id"],
                    "classification": result["summary"]["classification"],
                    "trade_count": float(metrics.get("trade_count", 0.0)),
                    "tpm": float(metrics.get("tpm", 0.0)),
                    "zero_trade_pct": float(metrics.get("zero_trade_pct", 100.0)),
                    "walkforward_executed_true_pct": float(metrics.get("walkforward_executed_true_pct", 0.0)),
                    "mc_trigger_rate": float(metrics.get("mc_trigger_rate", 0.0)),
                }
            )
    table = pd.DataFrame(rows)
    score_rows = []
    for horizon, group in table.groupby("horizon_months", sort=True):
        weak_or_better = group["classification"].isin({"ROBUST_EDGE", "WEAK_EDGE"}).mean()
        recency_overfit = float(group.sort_values("offset_months")["classification"].iloc[0] in {"ROBUST_EDGE", "WEAK_EDGE"} and group["classification"].isin({"ROBUST_EDGE", "WEAK_EDGE"}).sum() == 1)
        score = float(max(0.0, weak_or_better - 0.2 * recency_overfit))
        score_rows.append({"horizon_months": int(horizon), "horizon_consistency_score": score})
    score_df = pd.DataFrame(score_rows)
    overall = float(score_df["horizon_consistency_score"].mean()) if not score_df.empty else 0.0
    classification = "ROBUST_EDGE" if overall >= 0.6 else "WEAK_EDGE" if overall >= 0.3 else "NO_EDGE"

    payload = {
        "git_commit": "",
        "stage": str(stage_tag),
        "families": families,
        "composer_mode": composer_mode,
        "horizon_consistency_score": overall,
        "classification": classification,
        "warnings": [] if overall > 0 else ["insufficient_horizon_consistency"],
    }
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_json = docs_dir / f"{report_name}_summary.json"
    report_md = docs_dir / f"{report_name}_report.md"
    table_path = docs_dir / f"{report_name}_table.csv"
    score_path = docs_dir / f"{report_name}_scores.csv"
    table.to_csv(table_path, index=False)
    score_df.to_csv(score_path, index=False)
    report_json.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        f"# Stage-{stage_tag} Multi-Horizon Report",
        "",
        "## 1) What changed",
        "- Added rolling 3m/6m/12m non-overlapping horizon checks.",
        "",
        "## 2) How to run (dry-run + real)",
        f"- dry-run: `python scripts/run_stage13.py --substage {stage_tag} --dry-run --seed 42`",
        f"- real: `python scripts/run_stage13.py --substage {stage_tag} --seed 42`",
        "",
        "## 3) Validation gates & results",
        f"- horizon_consistency_score: `{overall:.6f}`",
        f"- classification: `{classification}`",
        "",
        "## 4) Key metrics tables (trade_count, tpm, PF, expectancy, exp_lcb, maxDD, wf, mc)",
        f"- windows table: `{table_path.as_posix()}`",
        f"- horizon scores: `{score_path.as_posix()}`",
        "",
        "## 5) Failures + reasons",
    ]
    if payload["warnings"]:
        lines.extend([f"- {item}" for item in payload["warnings"]])
    else:
        lines.append("- none")
    lines.extend(["", "## 6) Next actions", "- Feed stable horizons into Stage-14 ML-lite calibration."])
    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return {
        "summary": payload,
        "table": table,
        "scores": score_df,
        "report_md": report_md,
        "report_json": report_json,
    }
