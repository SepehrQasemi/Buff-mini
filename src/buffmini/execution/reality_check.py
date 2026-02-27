"""Stage-4.5 reality-check robustness layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.constants import RUNS_DIR


PASS_THRESHOLD = 0.70
WARN_THRESHOLD = 0.40


@dataclass(frozen=True)
class RealityCheckConfig:
    """Config for deterministic reality-check evaluation."""

    seed: int = 42
    weekly_step_days: int = 7
    perturbation_noise_bps: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0)
    execution_drag_cases: tuple[tuple[int, float], ...] = (
        (0, 0.0),
        (1, 1.0),
        (1, 2.0),
        (2, 3.0),
    )


@dataclass
class RealityCheckResult:
    """Reality-check payload and output tables."""

    summary: dict[str, Any]
    rolling_forward: pd.DataFrame
    perturbation: pd.DataFrame
    execution_drag: pd.DataFrame


def run_reality_check(
    run_id: str,
    runs_dir: Path = RUNS_DIR,
    cfg: RealityCheckConfig | None = None,
    policy_snapshot_path: Path | None = None,
) -> Path:
    """Run Stage-4.5 on one pipeline run and persist artifacts."""

    config = cfg or RealityCheckConfig()
    run_dir = Path(runs_dir) / str(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"run not found: {run_id}")

    summary_ui = _read_json(run_dir / "ui_bundle" / "summary_ui.json")
    if not summary_ui:
        raise FileNotFoundError(f"missing ui_bundle summary: {run_dir / 'ui_bundle' / 'summary_ui.json'}")
    pipeline_summary = _read_json(run_dir / "pipeline_summary.json")

    resolved_policy_snapshot_path = _resolve_policy_snapshot_path(
        run_dir=run_dir,
        runs_dir=Path(runs_dir),
        summary_ui=summary_ui,
        pipeline_summary=pipeline_summary,
        explicit_path=policy_snapshot_path,
    )
    policy_snapshot = _read_json(resolved_policy_snapshot_path)
    if not policy_snapshot:
        raise FileNotFoundError(f"missing or invalid policy snapshot: {resolved_policy_snapshot_path}")

    mismatches = _policy_binding_mismatches(summary_ui=summary_ui, policy_snapshot=policy_snapshot)
    if mismatches:
        raise ValueError("policy snapshot mismatch: " + "; ".join(mismatches))

    equity_df = _read_csv(run_dir / "ui_bundle" / "equity_curve.csv")
    if equity_df.empty:
        raise FileNotFoundError(f"missing ui_bundle equity curve: {run_dir / 'ui_bundle' / 'equity_curve.csv'}")

    exposure_df = _read_csv(run_dir / "ui_bundle" / "exposure.csv")
    trades_df = _read_csv(run_dir / "ui_bundle" / "trades.csv")

    equity_series = _equity_series(equity_df)
    returns = equity_series.pct_change().fillna(0.0).astype(float)

    exposure = _exposure_series(exposure_df, index=returns.index)
    baseline_metrics = _compute_metrics(returns)

    rolling_df = _rolling_forward_emulation(
        returns=returns,
        trades_df=trades_df,
        step_days=int(config.weekly_step_days),
    )

    perturb_df = _perturbation_table(
        returns=returns,
        noise_bps_list=list(config.perturbation_noise_bps),
        seed=int(config.seed),
    )

    execution_df = _execution_drag_table(
        returns=returns,
        exposure=exposure,
        drag_cases=list(config.execution_drag_cases),
    )

    _assert_no_nan(rolling_df, "rolling_forward_steps")
    _assert_no_nan(perturb_df, "perturbation_table")
    _assert_no_nan(execution_df, "execution_drag_table")

    invariants = _evaluate_invariants(
        baseline_metrics=baseline_metrics,
        perturb_df=perturb_df,
        execution_df=execution_df,
        summary_ui=summary_ui,
    )

    confidence_score, verdict, reasons = _score_and_verdict(
        rolling_df=rolling_df,
        perturb_df=perturb_df,
        execution_df=execution_df,
        invariants=invariants,
    )

    result = RealityCheckResult(
        summary={
            "run_id": str(run_id),
            "status": str(summary_ui.get("status", "unknown")),
            "seed": int(config.seed),
            "confidence_score": float(confidence_score),
            "verdict": str(verdict),
            "reasons": reasons,
            "baseline_metrics": baseline_metrics,
            "rolling_forward": {
                "step_days": int(config.weekly_step_days),
                "step_count": int(len(rolling_df)),
                "min_expectancy": float(rolling_df["expectancy"].min()) if not rolling_df.empty else 0.0,
                "min_profit_factor": float(rolling_df["profit_factor"].min()) if not rolling_df.empty else 0.0,
                "max_drawdown": float(rolling_df["max_drawdown"].max()) if not rolling_df.empty else 0.0,
            },
            "perturbation": {
                "noise_bps": [float(v) for v in config.perturbation_noise_bps],
                "worst_return_pct": float(perturb_df["return_pct"].min()) if not perturb_df.empty else 0.0,
                "worst_profit_factor": float(perturb_df["profit_factor"].min()) if not perturb_df.empty else 0.0,
            },
            "execution_drag": {
                "cases": [
                    {"delay_bars": int(delay), "extra_slippage_bps": float(slip)}
                    for delay, slip in config.execution_drag_cases
                ],
                "worst_return_pct": float(execution_df["return_pct"].min()) if not execution_df.empty else 0.0,
                "worst_profit_factor": float(execution_df["profit_factor"].min()) if not execution_df.empty else 0.0,
            },
            "invariants": invariants,
            "config_hash": summary_ui.get("config_hash"),
            "data_hash": summary_ui.get("data_hash"),
            "seed_source": summary_ui.get("seed"),
            "policy_snapshot_path": str(resolved_policy_snapshot_path),
            "policy_binding_validated": True,
        },
        rolling_forward=rolling_df,
        perturbation=perturb_df,
        execution_drag=execution_df,
    )

    rc_dir = run_dir / "reality_check"
    rc_dir.mkdir(parents=True, exist_ok=True)

    result.rolling_forward.to_csv(rc_dir / "rolling_forward_steps.csv", index=False)
    result.perturbation.to_csv(rc_dir / "perturbation_table.csv", index=False)
    result.execution_drag.to_csv(rc_dir / "execution_drag_table.csv", index=False)

    _write_json(rc_dir / "reality_check_summary.json", result.summary)
    _write_report(
        path=rc_dir / "reality_check_report.md",
        summary=result.summary,
        rolling=result.rolling_forward,
        perturbation=result.perturbation,
        execution_drag=result.execution_drag,
    )

    return rc_dir


def _rolling_forward_emulation(
    returns: pd.Series,
    trades_df: pd.DataFrame,
    step_days: int,
) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(
            columns=[
                "step_idx",
                "start_ts",
                "end_ts",
                "profit_factor",
                "expectancy",
                "max_drawdown",
                "return_pct",
                "trade_count",
            ]
        )

    idx = pd.DatetimeIndex(returns.index)
    start = idx.min().floor("D")
    end = idx.max()
    step_delta = pd.Timedelta(days=max(step_days, 1))

    rows: list[dict[str, Any]] = []
    step_idx = 0
    cursor = start
    while cursor <= end:
        step_start = cursor
        step_end = min(cursor + step_delta, end + pd.Timedelta(seconds=1))
        mask = (idx >= step_start) & (idx < step_end)
        window_returns = returns.loc[mask]
        if window_returns.empty:
            cursor += step_delta
            continue
        metrics = _compute_metrics(window_returns)
        trade_count = _count_trades(trades_df, step_start, step_end)
        rows.append(
            {
                "step_idx": int(step_idx),
                "start_ts": _to_utc_iso(step_start),
                "end_ts": _to_utc_iso(step_end),
                "profit_factor": float(metrics["profit_factor"]),
                "expectancy": float(metrics["expectancy"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "return_pct": float(metrics["return_pct"]),
                "trade_count": int(trade_count),
            }
        )
        step_idx += 1
        cursor += step_delta

    return pd.DataFrame(rows)


def _perturbation_table(
    returns: pd.Series,
    noise_bps_list: list[float],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = _compute_metrics(returns)

    for noise_bps in noise_bps_list:
        if float(noise_bps) == 0.0:
            noisy = returns.copy()
        else:
            rng = np.random.default_rng(seed + int(float(noise_bps) * 1000) + 11)
            sigma = float(noise_bps) / 10_000.0
            noisy = (returns + rng.normal(0.0, sigma, size=len(returns))).astype(float)

        metrics = _compute_metrics(noisy)
        rows.append(
            {
                "noise_bps": float(noise_bps),
                "profit_factor": float(metrics["profit_factor"]),
                "expectancy": float(metrics["expectancy"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "return_pct": float(metrics["return_pct"]),
                "trade_count": int(metrics["trade_count"]),
                "delta_return_vs_base": float(metrics["return_pct"] - baseline["return_pct"]),
            }
        )

    return pd.DataFrame(rows)


def _execution_drag_table(
    returns: pd.Series,
    exposure: pd.Series,
    drag_cases: list[tuple[int, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = _compute_metrics(returns)

    for delay_bars, extra_slippage_bps in drag_cases:
        if int(delay_bars) <= 0:
            delayed = returns.copy()
        else:
            delayed = returns.shift(int(delay_bars)).fillna(0.0)

        slip_per_bar = (float(extra_slippage_bps) / 10_000.0) * exposure.abs()
        dragged = (delayed - slip_per_bar).astype(float)

        metrics = _compute_metrics(dragged)
        rows.append(
            {
                "delay_bars": int(delay_bars),
                "extra_slippage_bps": float(extra_slippage_bps),
                "profit_factor": float(metrics["profit_factor"]),
                "expectancy": float(metrics["expectancy"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "return_pct": float(metrics["return_pct"]),
                "trade_count": int(metrics["trade_count"]),
                "delta_return_vs_base": float(metrics["return_pct"] - baseline["return_pct"]),
            }
        )

    return pd.DataFrame(rows)


def _evaluate_invariants(
    baseline_metrics: dict[str, float],
    perturb_df: pd.DataFrame,
    execution_df: pd.DataFrame,
    summary_ui: dict[str, Any],
) -> dict[str, Any]:
    perturb_base = perturb_df.loc[perturb_df["noise_bps"] == 0.0]
    drag_base = execution_df.loc[(execution_df["delay_bars"] == 0) & (execution_df["extra_slippage_bps"] == 0.0)]

    perturb_baseline_match = False
    drag_baseline_match = False
    if not perturb_base.empty:
        row = perturb_base.iloc[0]
        perturb_baseline_match = (
            abs(float(row["return_pct"]) - float(baseline_metrics["return_pct"])) <= 1e-12
            and abs(float(row["max_drawdown"]) - float(baseline_metrics["max_drawdown"])) <= 1e-12
            and abs(float(row["expectancy"]) - float(baseline_metrics["expectancy"])) <= 1e-12
        )
    if not drag_base.empty:
        row = drag_base.iloc[0]
        drag_baseline_match = (
            abs(float(row["return_pct"]) - float(baseline_metrics["return_pct"])) <= 1e-12
            and abs(float(row["max_drawdown"]) - float(baseline_metrics["max_drawdown"])) <= 1e-12
            and abs(float(row["expectancy"]) - float(baseline_metrics["expectancy"])) <= 1e-12
        )

    summary_pf = (summary_ui.get("key_metrics") or {}).get("pf")
    summary_dd = (summary_ui.get("key_metrics") or {}).get("maxdd")

    return {
        "confidence_score_range_ok": True,
        "verdict_threshold_alignment_ok": True,
        "perturbation_baseline_matches_original": bool(perturb_baseline_match),
        "execution_drag_baseline_matches_original": bool(drag_baseline_match),
        "summary_vs_baseline_pf_delta": (
            abs(float(summary_pf) - float(baseline_metrics["profit_factor"])) if summary_pf is not None else None
        ),
        "summary_vs_baseline_maxdd_delta": (
            abs(float(summary_dd) - float(baseline_metrics["max_drawdown"])) if summary_dd is not None else None
        ),
    }


def _score_and_verdict(
    rolling_df: pd.DataFrame,
    perturb_df: pd.DataFrame,
    execution_df: pd.DataFrame,
    invariants: dict[str, Any],
) -> tuple[float, str, list[str]]:
    reasons: list[str] = []

    if rolling_df.empty:
        roll_score = 0.0
    else:
        stable_steps = (
            (pd.to_numeric(rolling_df["expectancy"], errors="coerce") >= 0.0)
            & (pd.to_numeric(rolling_df["profit_factor"], errors="coerce") >= 1.0)
        ).mean()
        roll_score = float(stable_steps)

    base_return = float(perturb_df.loc[perturb_df["noise_bps"] == 0.0, "return_pct"].iloc[0]) if not perturb_df.empty else 0.0

    worst_pert = float(perturb_df["return_pct"].min()) if not perturb_df.empty else 0.0
    worst_drag = float(execution_df["return_pct"].min()) if not execution_df.empty else 0.0

    if base_return > 0:
        pert_score = float(np.clip(worst_pert / base_return, 0.0, 1.0))
        drag_score = float(np.clip(worst_drag / base_return, 0.0, 1.0))
    else:
        pert_score = 0.0 if worst_pert < 0 else 0.5
        drag_score = 0.0 if worst_drag < 0 else 0.5

    confidence_score = float(np.clip((0.4 * roll_score) + (0.3 * pert_score) + (0.3 * drag_score), 0.0, 1.0))

    if confidence_score >= PASS_THRESHOLD:
        verdict = "PASS"
    elif confidence_score >= WARN_THRESHOLD:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    if roll_score < 0.5:
        reasons.append("Low weekly forward-step stability")
    if pert_score < 0.5:
        reasons.append("High sensitivity to data perturbation noise")
    if drag_score < 0.5:
        reasons.append("High sensitivity to execution drag")
    if not bool(invariants.get("perturbation_baseline_matches_original")):
        reasons.append("Perturbation baseline does not match original metrics")
    if not bool(invariants.get("execution_drag_baseline_matches_original")):
        reasons.append("Execution-drag baseline does not match original metrics")

    if not reasons:
        reasons.append("No critical fragility observed under configured stress grid")

    return confidence_score, verdict, reasons


def _compute_metrics(returns: pd.Series) -> dict[str, float]:
    ret = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if ret.empty:
        return {
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "return_pct": 0.0,
            "trade_count": 0.0,
        }

    positive = ret[ret > 0.0]
    negative = ret[ret < 0.0]
    gross_profit = float(positive.sum())
    gross_loss = float(-negative.sum())
    if gross_loss <= 0:
        profit_factor = 10.0 if gross_profit > 0 else 0.0
    else:
        profit_factor = float(gross_profit / gross_loss)
    profit_factor = float(np.clip(profit_factor, 0.0, 10.0))

    active = ret[ret != 0.0]
    expectancy = float(active.mean()) if not active.empty else 0.0

    equity = (1.0 + ret).cumprod()
    running_peak = equity.cummax().replace(0.0, np.nan)
    drawdown = ((running_peak - equity) / running_peak).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_drawdown = float(np.clip(drawdown.max(), 0.0, 1.0))

    return_pct = float(equity.iloc[-1] - 1.0)
    trade_count = int((ret != 0.0).sum())

    return {
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "max_drawdown": float(max_drawdown),
        "return_pct": float(return_pct),
        "trade_count": float(trade_count),
    }


def _equity_series(equity_df: pd.DataFrame) -> pd.Series:
    frame = equity_df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
    if frame.empty:
        raise ValueError("equity_curve.csv has no valid rows")
    return pd.Series(frame["equity"].astype(float).values, index=pd.DatetimeIndex(frame["timestamp"], name="timestamp"))


def _exposure_series(exposure_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    if exposure_df.empty or "timestamp" not in exposure_df.columns:
        return pd.Series(1.0, index=index, dtype=float)

    frame = exposure_df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    exposure_col = "exposure" if "exposure" in frame.columns else "gross_exposure"
    if exposure_col not in frame.columns:
        return pd.Series(1.0, index=index, dtype=float)

    frame[exposure_col] = pd.to_numeric(frame[exposure_col], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        return pd.Series(1.0, index=index, dtype=float)

    aligned = frame.set_index("timestamp")[exposure_col].reindex(index).ffill().fillna(0.0)
    # Keep a minimum activity proxy to avoid zero-cost artifacts.
    return aligned.abs().clip(lower=0.25).astype(float)


def _count_trades(trades_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    if trades_df.empty:
        return 0

    frame = trades_df.copy()
    ts_col = None
    for col in ["timestamp", "ts", "entry_time", "entry_ts"]:
        if col in frame.columns:
            ts_col = col
            break
    if ts_col is None:
        return 0

    frame["_ts"] = pd.to_datetime(frame[ts_col], utc=True, errors="coerce")
    frame = frame.dropna(subset=["_ts"])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    mask = (frame["_ts"] >= start_ts) & (frame["_ts"] < end_ts)
    return int(mask.sum())


def _to_utc_iso(value: pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _assert_no_nan(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        return
    numeric = frame.select_dtypes(include=["number"])
    if numeric.isna().any().any():
        raise ValueError(f"{label} contains NaN values")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError(f"{label} contains infinite values")


def _write_report(
    path: Path,
    summary: dict[str, Any],
    rolling: pd.DataFrame,
    perturbation: pd.DataFrame,
    execution_drag: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Stage-4.5 Reality Check Report")
    lines.append("")
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- confidence_score: `{summary['confidence_score']:.4f}`")
    lines.append(f"- verdict: `{summary['verdict']}`")
    lines.append(f"- reasons: `{summary['reasons']}`")
    lines.append("")

    lines.append("## Baseline Metrics")
    lines.append(f"- {summary['baseline_metrics']}")
    lines.append("")

    lines.append("## Rolling Forward (Weekly)")
    lines.append(f"- step_count: `{summary['rolling_forward']['step_count']}`")
    lines.append(f"- min_expectancy: `{summary['rolling_forward']['min_expectancy']:.6f}`")
    lines.append(f"- min_profit_factor: `{summary['rolling_forward']['min_profit_factor']:.6f}`")
    lines.append("")
    if not rolling.empty:
        lines.append("| step_idx | start_ts | end_ts | pf | expectancy | max_dd | return_pct | trade_count |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in rolling.to_dict(orient="records"):
            lines.append(
                f"| {row['step_idx']} | {row['start_ts']} | {row['end_ts']} | "
                f"{float(row['profit_factor']):.4f} | {float(row['expectancy']):.6f} | "
                f"{float(row['max_drawdown']):.4f} | {float(row['return_pct']):.4f} | {int(row['trade_count'])} |"
            )
        lines.append("")

    lines.append("## Perturbation")
    if not perturbation.empty:
        lines.append("| noise_bps | pf | expectancy | max_dd | return_pct | delta_return_vs_base |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in perturbation.to_dict(orient="records"):
            lines.append(
                f"| {float(row['noise_bps']):.2f} | {float(row['profit_factor']):.4f} | {float(row['expectancy']):.6f} | "
                f"{float(row['max_drawdown']):.4f} | {float(row['return_pct']):.4f} | {float(row['delta_return_vs_base']):.4f} |"
            )
        lines.append("")

    lines.append("## Execution Drag")
    if not execution_drag.empty:
        lines.append("| delay_bars | extra_slippage_bps | pf | expectancy | max_dd | return_pct | delta_return_vs_base |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in execution_drag.to_dict(orient="records"):
            lines.append(
                f"| {int(row['delay_bars'])} | {float(row['extra_slippage_bps']):.2f} | {float(row['profit_factor']):.4f} | "
                f"{float(row['expectancy']):.6f} | {float(row['max_drawdown']):.4f} | {float(row['return_pct']):.4f} | "
                f"{float(row['delta_return_vs_base']):.4f} |"
            )

    lines.append("")
    lines.append("## Invariants")
    lines.append(f"- {summary['invariants']}")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_policy_snapshot_path(
    run_dir: Path,
    runs_dir: Path,
    summary_ui: dict[str, Any],
    pipeline_summary: dict[str, Any],
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)

    stage4_run_id = str(
        summary_ui.get("stages", {}).get("stage4_run_id")
        or pipeline_summary.get("stage4_run_id")
        or ""
    ).strip()
    if stage4_run_id:
        return runs_dir / stage4_run_id / "policy_snapshot.json"

    fallback = run_dir / "policy_snapshot.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Stage-4 policy snapshot could not be resolved from run metadata")


def _policy_binding_mismatches(summary_ui: dict[str, Any], policy_snapshot: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    ui_policy = summary_ui.get("policy_snapshot")
    if not isinstance(ui_policy, dict):
        return ["summary_ui.policy_snapshot missing"]

    top_checks = [
        ("chosen_leverage", ("leverage",)),
        ("execution_mode", ("execution_mode",)),
    ]
    for ui_key, snap_path in top_checks:
        ui_val = summary_ui.get(ui_key)
        snap_val = _nested_get(policy_snapshot, *snap_path)
        if not _values_equal(ui_val, snap_val):
            mismatches.append(f"{ui_key} != policy_snapshot.{'.'.join(snap_path)}")

    nested_checks = [
        ("leverage",),
        ("execution_mode",),
        ("caps", "max_gross_exposure"),
        ("caps", "max_net_exposure_per_symbol"),
        ("caps", "max_open_positions"),
        ("costs", "round_trip_cost_pct"),
        ("costs", "slippage_pct"),
        ("costs", "funding_pct_per_day"),
        ("kill_switch", "enabled"),
        ("kill_switch", "max_daily_loss_pct"),
        ("kill_switch", "max_peak_to_valley_dd_pct"),
        ("kill_switch", "max_consecutive_losses"),
        ("kill_switch", "cool_down_bars"),
        ("stage6", "stage6_enabled"),
        ("stage6", "regime_rules", "atr_percentile_window"),
        ("stage6", "regime_rules", "vol_expansion_threshold"),
        ("stage6", "regime_rules", "trend_strength_threshold"),
        ("stage6", "regime_rules", "range_atr_threshold"),
        ("stage6", "dynamic_leverage", "trend_multiplier"),
        ("stage6", "dynamic_leverage", "range_multiplier"),
        ("stage6", "dynamic_leverage", "vol_expansion_multiplier"),
        ("stage6", "dynamic_leverage", "dd_soft_threshold"),
        ("stage6", "dynamic_leverage", "dd_soft_multiplier"),
        ("stage6", "dynamic_leverage", "max_leverage"),
    ]
    for path in nested_checks:
        ui_val = _nested_get(ui_policy, *path)
        snap_val = _nested_get(policy_snapshot, *path)
        if not _values_equal(ui_val, snap_val):
            joined = ".".join(path)
            mismatches.append(f"summary_ui.policy_snapshot.{joined} != policy_snapshot.{joined}")

    return mismatches


def _nested_get(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _values_equal(left: Any, right: Any) -> bool:
    try:
        left_num = float(left)
        right_num = float(right)
        return abs(left_num - right_num) <= 1e-12
    except Exception:
        return left == right


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
