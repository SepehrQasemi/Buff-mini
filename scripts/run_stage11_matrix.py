"""Run Stage-11.1 baseline/preset matrix and write comparative report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.storage import parquet_path
from buffmini.stage11.evaluate import apply_stage11_preset, run_stage11


PRESET_FILES = {
    "bias": Path("configs/presets/stage11_bias.yaml"),
    "confirm": Path("configs/presets/stage11_confirm.yaml"),
    "bias_confirm": Path("configs/presets/stage11_bias_confirm.yaml"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-11.1 effectiveness matrix")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--window-months", type=int, default=3)
    parser.add_argument("--real", dest="real", action="store_true", default=None)
    parser.add_argument("--no-real", dest="real", action="store_false")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--allow-noop", action="store_true", help="Allow NO-OP BUG runs")
    return parser.parse_args()


def run_stage11_matrix(
    *,
    config: dict[str, Any],
    seed: int,
    symbols: list[str],
    timeframe: str,
    default_window_months: int,
    run_real: bool,
    runs_dir: Path,
    docs_dir: Path,
    data_dir: Path,
    derived_dir: Path,
    allow_noop: bool = False,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    notes: list[str] = []

    synthetic_windows = [int(default_window_months)]
    for months in synthetic_windows:
        results.extend(
            _run_dataset_modes(
                base_config=config,
                dataset_name="synthetic",
                dry_run=True,
                window_months=int(months),
                seed=int(seed),
                symbols=symbols,
                timeframe=timeframe,
                runs_dir=runs_dir,
                docs_dir=docs_dir,
                data_dir=data_dir,
                derived_dir=derived_dir,
                allow_noop=allow_noop,
            )
        )

    real_available = _real_data_available(symbols=symbols, timeframe=timeframe, data_dir=data_dir)
    coverage_days = _min_real_coverage_days(symbols=symbols, timeframe=timeframe, data_dir=data_dir) if real_available else 0.0
    do_real = bool(real_available and (run_real or run_real is None))
    if not real_available:
        notes.append("real_data_unavailable_for_requested_symbols")
    elif run_real is False:
        notes.append("real_data_run_skipped_by_flag")
    if do_real:
        real_windows = [int(default_window_months)]
        if int(default_window_months) != 12 and coverage_days >= 330.0:
            real_windows.append(12)
        elif coverage_days < 330.0:
            notes.append(f"real_data_12m_window_skipped_insufficient_coverage_days={coverage_days:.1f}")
        for months in real_windows:
            results.extend(
                _run_dataset_modes(
                    base_config=config,
                    dataset_name="real",
                    dry_run=False,
                    window_months=int(months),
                    seed=int(seed),
                    symbols=symbols,
                    timeframe=timeframe,
                    runs_dir=runs_dir,
                    docs_dir=docs_dir,
                    data_dir=data_dir,
                    derived_dir=derived_dir,
                    allow_noop=allow_noop,
                )
            )

    if not results:
        raise ValueError("Stage-11.1 matrix produced no results")

    best_mode = {
        "synthetic": _best_mode(results, dataset_name="synthetic"),
        "real": _best_mode(results, dataset_name="real"),
    }
    noop_detected = any(bool(row.get("noop_bug_detected", False)) for row in results)
    summary = {
        "stage": "11.1",
        "seed": int(seed),
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "modes": results,
        "best_mode": best_mode,
        "notes": notes,
        "noop_detected": bool(noop_detected),
    }
    validate_stage11_1_summary_schema(summary)
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "stage11_1_report_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_stage11_1_report(summary=summary, out_path=docs_dir / "stage11_1_report.md")
    return summary


def validate_stage11_1_summary_schema(summary: dict[str, Any]) -> None:
    required = {"stage", "seed", "generated_at_utc", "modes", "best_mode", "notes", "noop_detected"}
    missing = required.difference(summary.keys())
    if missing:
        raise ValueError(f"Missing Stage-11.1 summary keys: {sorted(missing)}")
    if str(summary["stage"]) != "11.1":
        raise ValueError("stage must be '11.1'")
    modes = summary.get("modes", [])
    if not isinstance(modes, list) or not modes:
        raise ValueError("modes must be non-empty list")
    required_mode_keys = {
        "name",
        "dataset",
        "window_months",
        "run_id",
        "verdict",
        "metrics",
        "deltas",
        "walkforward",
        "trade_count_guard",
        "noop_bug_detected",
    }
    for idx, row in enumerate(modes):
        if not isinstance(row, dict):
            raise ValueError(f"modes[{idx}] must be object")
        missing_mode = required_mode_keys.difference(row.keys())
        if missing_mode:
            raise ValueError(f"modes[{idx}] missing keys: {sorted(missing_mode)}")
        if str(row["name"]) not in {"baseline", "bias", "confirm", "bias_confirm"}:
            raise ValueError(f"modes[{idx}].name invalid")
        if str(row["dataset"]) not in {"synthetic", "real"}:
            raise ValueError(f"modes[{idx}].dataset invalid")


def _run_dataset_modes(
    *,
    base_config: dict[str, Any],
    dataset_name: str,
    dry_run: bool,
    window_months: int,
    seed: int,
    symbols: list[str],
    timeframe: str,
    runs_dir: Path,
    docs_dir: Path,
    data_dir: Path,
    derived_dir: Path,
    allow_noop: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    guard_cfg = (
        base_config.get("evaluation", {})
        .get("stage11", {})
        .get("trade_count_guard", {})
    )
    bias_threshold = float(guard_cfg.get("bias_max_drop_pct", 2.0))
    confirm_threshold = float(guard_cfg.get("confirm_max_drop_pct", 25.0))

    cfg_confirm = apply_stage11_preset(json.loads(json.dumps(base_config)), PRESET_FILES["confirm"])
    summary_confirm = run_stage11(
        config=cfg_confirm,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=symbols,
        timeframe=timeframe,
        cost_mode="v2",
        walkforward_v2_enabled=True,
        runs_root=runs_dir,
        docs_dir=docs_dir,
        data_dir=data_dir,
        derived_dir=derived_dir,
        write_docs=False,
        allow_noop=bool(allow_noop),
        window_months=int(window_months),
    )

    cfg_bias_confirm = apply_stage11_preset(json.loads(json.dumps(base_config)), PRESET_FILES["bias_confirm"])
    summary_bias_confirm = run_stage11(
        config=cfg_bias_confirm,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=symbols,
        timeframe=timeframe,
        cost_mode="v2",
        walkforward_v2_enabled=True,
        runs_root=runs_dir,
        docs_dir=docs_dir,
        data_dir=data_dir,
        derived_dir=derived_dir,
        write_docs=False,
        allow_noop=bool(allow_noop),
        window_months=int(window_months),
    )

    baseline_metrics = _metrics_for_mode(summary_confirm, mode_name="baseline")
    mode_payloads = {
        "baseline": {
            "summary": summary_confirm,
            "metrics": baseline_metrics,
            "guard_max_drop": float(confirm_threshold),
            "noop": False,
        },
        "bias": {
            "summary": summary_bias_confirm,
            "metrics": _metrics_for_mode(summary_bias_confirm, mode_name="bias"),
            "guard_max_drop": float(bias_threshold),
            "noop": bool(float(summary_bias_confirm.get("sizing_stats", {}).get("pct_not_1_0", 0.0)) == 0.0),
        },
        "confirm": {
            "summary": summary_confirm,
            "metrics": _metrics_for_mode(summary_confirm, mode_name="confirm"),
            "guard_max_drop": float(confirm_threshold),
            "noop": bool(summary_confirm.get("noop_bug_detected", False)),
        },
        "bias_confirm": {
            "summary": summary_bias_confirm,
            "metrics": _metrics_for_mode(summary_bias_confirm, mode_name="bias_confirm"),
            "guard_max_drop": float(confirm_threshold),
            "noop": bool(summary_bias_confirm.get("noop_bug_detected", False)),
        },
    }

    for mode_name in ["baseline", "bias", "confirm", "bias_confirm"]:
        payload = mode_payloads[mode_name]
        summary = payload["summary"]
        metrics = payload["metrics"]
        deltas = {
            "trade_count_delta_pct": _delta_pct(metrics.get("trade_count", 0.0), baseline_metrics.get("trade_count", 0.0)),
            "profit_factor_delta": float(metrics.get("profit_factor", 0.0) - baseline_metrics.get("profit_factor", 0.0)),
            "expectancy_delta": float(metrics.get("expectancy", 0.0) - baseline_metrics.get("expectancy", 0.0)),
            "max_drawdown_delta": float(metrics.get("max_drawdown", 0.0) - baseline_metrics.get("max_drawdown", 0.0)),
            "exp_lcb_delta": float(metrics.get("exp_lcb", 0.0) - baseline_metrics.get("exp_lcb", 0.0)),
        }
        observed_drop = max(0.0, -float(deltas["trade_count_delta_pct"]))
        guard_pass = bool(observed_drop <= float(payload["guard_max_drop"]))
        walkforward = _walkforward_for_mode(summary=summary, mode_name=mode_name)
        verdict = _mode_verdict(
            mode_name=mode_name,
            guard_pass=guard_pass,
            noop_bug=bool(payload["noop"]),
            pf_delta=float(deltas["profit_factor_delta"]),
            expectancy_delta=float(deltas["expectancy_delta"]),
        )
        row = {
            "name": mode_name,
            "dataset": dataset_name,
            "window_months": int(window_months),
            "run_id": str(summary.get("run_id", "")),
            "verdict": verdict,
            "metrics": metrics,
            "deltas": deltas,
            "walkforward": walkforward,
            "trade_count_guard": {
                "max_drop_pct": float(payload["guard_max_drop"]),
                "observed_drop_pct": float(observed_drop),
                "pass": bool(guard_pass),
            },
            "noop_bug_detected": bool(payload["noop"]),
        }
        if mode_name in {"bias", "bias_confirm"}:
            row["sizing_stats"] = dict(summary.get("sizing_stats", {}))
        if mode_name in {"confirm", "bias_confirm"}:
            row["confirm_stats"] = dict(summary.get("confirm_stats", {}))
            row["entry_delta"] = dict(summary.get("entry_delta", {}))
        out.append(row)
    return out


def _metrics_for_mode(summary: dict[str, Any], mode_name: str) -> dict[str, float]:
    comp = dict(summary.get("comparisons", {}))
    if mode_name == "baseline":
        raw = dict(comp.get("baseline_stage10_7", {}))
    elif mode_name == "bias":
        raw = dict(comp.get("stage11_bias_only", {}))
    else:
        raw = dict(comp.get("stage11_with_confirm", {}))
    window_months = float(summary.get("window_months") or 1.0)
    trade_count = float(raw.get("trade_count", 0.0))
    return {
        "trade_count": trade_count,
        "profit_factor": float(raw.get("profit_factor", 0.0)),
        "expectancy": float(raw.get("expectancy", 0.0)),
        "max_drawdown": float(raw.get("max_drawdown", 0.0)),
        "exp_lcb": float(raw.get("exp_lcb", 0.0)),
        "tpm": float(trade_count / max(window_months, 1.0)),
    }


def _walkforward_for_mode(summary: dict[str, Any], mode_name: str) -> dict[str, Any]:
    walk = dict(summary.get("walkforward", {}))
    if mode_name == "baseline":
        classification = str(walk.get("baseline_classification", "N/A"))
        usable = int(walk.get("baseline_usable_windows", 0))
    else:
        classification = str(walk.get("stage11_classification", "N/A"))
        usable = int(walk.get("stage11_usable_windows", 0))
    return {"classification": classification, "usable_windows": usable}


def _delta_pct(candidate: float, baseline: float) -> float:
    b = float(baseline)
    c = float(candidate)
    if b == 0:
        return 0.0
    return float((c - b) / b * 100.0)


def _best_mode(rows: list[dict[str, Any]], dataset_name: str) -> str:
    candidates = [
        row
        for row in rows
        if str(row.get("dataset")) == dataset_name
        and str(row.get("name")) != "baseline"
        and not bool(row.get("noop_bug_detected", False))
        and bool((row.get("trade_count_guard") or {}).get("pass", False))
    ]
    if not candidates:
        return "baseline"

    def _key(row: dict[str, Any]) -> tuple[float, float, float]:
        deltas = row.get("deltas") or {}
        metrics = row.get("metrics") or {}
        return (
            float(deltas.get("exp_lcb_delta", 0.0)),
            float(deltas.get("profit_factor_delta", 0.0)),
            -float(metrics.get("max_drawdown", 0.0)),
        )

    best = sorted(candidates, key=_key, reverse=True)[0]
    return str(best.get("name", "baseline"))


def _mode_verdict(mode_name: str, guard_pass: bool, noop_bug: bool, pf_delta: float, expectancy_delta: float) -> str:
    if bool(noop_bug):
        return "NO-OP BUG"
    if not bool(guard_pass):
        return "REGRESSION"
    if str(mode_name) == "baseline":
        return "NEUTRAL"
    if float(pf_delta) > 0.0 and float(expectancy_delta) > 0.0:
        return "IMPROVEMENT"
    return "NEUTRAL"


def _real_data_available(symbols: list[str], timeframe: str, data_dir: Path) -> bool:
    return all(parquet_path(symbol=symbol, timeframe=timeframe, data_dir=data_dir).exists() for symbol in symbols)


def _min_real_coverage_days(symbols: list[str], timeframe: str, data_dir: Path) -> float:
    coverages: list[float] = []
    for symbol in symbols:
        path = parquet_path(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
        frame = pd.read_parquet(path, columns=["timestamp"])
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        if ts.empty:
            continue
        coverage = float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0)
        coverages.append(coverage)
    return min(coverages) if coverages else 0.0


def _write_stage11_1_report(summary: dict[str, Any], out_path: Path) -> None:
    rows = list(summary.get("modes", []))
    lines: list[str] = []
    lines.append("# Stage-11.1 Effectiveness Report")
    lines.append("")
    lines.append("## 1) What changed vs prior Stage-11")
    lines.append("- Stage-11.1 enables explicit presets for bias/confirm/bias+confirm and enforces no-op detection.")
    lines.append("- Runs fail with `NO-OP BUG` when hooks are enabled but do not cause measurable effects.")
    lines.append("")
    lines.append("## 2) Presets")
    lines.append("- `stage11_bias`: 4h context bias only, multiplier clamp 0.85..1.15.")
    lines.append("- `stage11_confirm`: 15m confirm only, delay up to 3 base bars, deterministic thresholding.")
    lines.append("- `stage11_bias_confirm`: both hooks enabled.")
    lines.append("")
    lines.append("## 3) Mode Results")
    lines.append("| dataset | window_months | mode | verdict | trade_count | PF | expectancy | exp_lcb | maxDD | tpm | trade_count_delta_pct | guard_pass | wf_class | wf_usable |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|")
    for row in rows:
        metrics = row.get("metrics") or {}
        deltas = row.get("deltas") or {}
        guard = row.get("trade_count_guard") or {}
        walk = row.get("walkforward") or {}
        lines.append(
            f"| {row.get('dataset')} | {int(row.get('window_months', 0))} | {row.get('name')} | {row.get('verdict')} | "
            f"{float(metrics.get('trade_count', 0.0)):.1f} | {float(metrics.get('profit_factor', 0.0)):.6f} | "
            f"{float(metrics.get('expectancy', 0.0)):.6f} | {float(metrics.get('exp_lcb', 0.0)):.6f} | "
            f"{float(metrics.get('max_drawdown', 0.0)):.6f} | {float(metrics.get('tpm', 0.0)):.3f} | "
            f"{float(deltas.get('trade_count_delta_pct', 0.0)):.3f}% | {bool(guard.get('pass', False))} | "
            f"{walk.get('classification', 'N/A')} | {int(walk.get('usable_windows', 0))} |"
        )
    lines.append("")
    lines.append("## 4) Effectiveness Proof")
    for row in rows:
        if row.get("name") == "bias":
            s = row.get("sizing_stats") or {}
            lines.append(
                f"- `{row.get('dataset')}` bias sizing stats: mean={float(s.get('mean_multiplier', 1.0)):.6f}, "
                f"p05={float(s.get('p05', 1.0)):.6f}, p95={float(s.get('p95', 1.0)):.6f}, "
                f"pct_not_1.0={float(s.get('pct_not_1_0', 0.0)):.3f}"
            )
        if row.get("name") in {"confirm", "bias_confirm"}:
            c = row.get("confirm_stats") or {}
            lines.append(
                f"- `{row.get('dataset')}` {row.get('name')} confirm stats: seen={int(c.get('signals_seen', 0))}, "
                f"confirmed={int(c.get('confirmed', 0))}, skipped={int(c.get('skipped', 0))}, "
                f"confirm_rate={float(c.get('confirm_rate', 0.0)):.3f}, median_delay={float(c.get('median_delay_bars', 0.0)):.2f}"
            )
    lines.append("")
    lines.append("## 5) Best Mode")
    lines.append(f"- Synthetic best mode: `{summary.get('best_mode', {}).get('synthetic', 'baseline')}`")
    lines.append(f"- Real best mode: `{summary.get('best_mode', {}).get('real', 'baseline')}`")
    lines.append("")
    lines.append("## 6) Notes")
    notes = summary.get("notes") or []
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- none")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    run_real = args.real if args.real is not None else True
    summary = run_stage11_matrix(
        config=config,
        seed=int(args.seed),
        symbols=symbols,
        timeframe=str(args.timeframe),
        default_window_months=int(args.window_months),
        run_real=bool(run_real),
        runs_dir=args.runs_dir,
        docs_dir=args.docs_dir,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
        allow_noop=bool(args.allow_noop),
    )
    print(f"stage: {summary['stage']}")
    print(f"modes_run: {len(summary['modes'])}")
    print(f"best_mode.synthetic: {summary['best_mode']['synthetic']}")
    print(f"best_mode.real: {summary['best_mode']['real']}")
    print("wrote: docs/stage11_1_report.md")
    print("wrote: docs/stage11_1_report_summary.json")


if __name__ == "__main__":
    main()
