"""Build standardized ui_bundle artifacts for Stage-5 pages."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from buffmini.config import compute_config_hash
from buffmini.constants import PROJECT_ROOT, RUNS_DIR
from buffmini.ui_bundle.discover import find_best_report_files, find_frontier_selector_tables


def build_ui_bundle(run_id: str, source_paths: dict[str, Any], out_dir: Path) -> None:
    """Build ui_bundle from discovered source paths.

    Parameters
    ----------
    run_id:
        Pipeline run id.
    source_paths:
        Pre-discovered source files/dirs and metadata.
    out_dir:
        Target ui_bundle directory.
    """

    warnings: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_run_dir = Path(source_paths.get("pipeline_run_dir", RUNS_DIR / run_id))
    pipeline_summary = _safe_json(Path(source_paths.get("pipeline_summary_path", pipeline_run_dir / "pipeline_summary.json")))
    progress = _safe_json(pipeline_run_dir / "progress.json")

    pipeline_cfg_path = Path(source_paths.get("pipeline_config_path", pipeline_run_dir / "pipeline_config.yaml"))
    pipeline_cfg = _safe_yaml(pipeline_cfg_path)

    stage1_run_id = str(pipeline_summary.get("stage1_run_id") or "")
    stage2_run_id = str(pipeline_summary.get("stage2_run_id") or "")
    stage3_2_run_id = str(pipeline_summary.get("stage3_2_run_id") or "")
    stage3_3_run_id = str(pipeline_summary.get("stage3_3_run_id") or "")
    stage4_run_id = str(pipeline_summary.get("stage4_run_id") or "")
    stage4_sim_run_id = str(pipeline_summary.get("stage4_sim_run_id") or "")
    status = str(pipeline_summary.get("status") or progress.get("status") or "unknown")
    _validate_lineage_fields(
        status=status,
        stage1_run_id=stage1_run_id,
        stage2_run_id=stage2_run_id,
        stage3_3_run_id=stage3_3_run_id,
    )

    stage2_summary = _safe_json(RUNS_DIR / stage2_run_id / "portfolio_summary.json") if stage2_run_id else {}
    stage3_summary = _safe_json(RUNS_DIR / stage3_3_run_id / "selector_summary.json") if stage3_3_run_id else {}

    chosen_method = str(
        pipeline_summary.get("chosen_method")
        or (stage3_summary.get("overall_choice") or {}).get("method")
        or (pipeline_cfg.get("evaluation", {}).get("stage4", {}) or {}).get("default_method", "equal")
    )
    chosen_leverage = float(
        pipeline_summary.get("chosen_leverage")
        or (stage3_summary.get("overall_choice") or {}).get("chosen_leverage")
        or (pipeline_cfg.get("evaluation", {}).get("stage4", {}) or {}).get("default_leverage", 1.0)
    )
    execution_mode = str((pipeline_cfg.get("execution", {}) or {}).get("mode", "net"))

    symbols = list((pipeline_cfg.get("universe", {}) or {}).get("symbols", []))
    timeframe = str((pipeline_cfg.get("universe", {}) or {}).get("timeframe", "1h"))
    evaluation_window_months = int((pipeline_cfg.get("evaluation", {}).get("stage1", {}) or {}).get("holdout_months", 0) or 0)
    seed = int((pipeline_cfg.get("search", {}) or {}).get("seed", 42) or 42)
    resolved_end_ts = str(
        pipeline_summary.get("resolved_end_ts")
        or (pipeline_cfg.get("universe", {}) or {}).get("resolved_end_ts")
        or (pipeline_cfg.get("universe", {}) or {}).get("end")
        or ""
    )
    policy_snapshot = _load_policy_snapshot(
        stage4_run_id=stage4_run_id,
        pipeline_cfg=pipeline_cfg,
        chosen_method=chosen_method,
        chosen_leverage=chosen_leverage,
        warnings=warnings,
    )

    key_metrics = {
        "pf": None,
        "maxdd": None,
        "p_ruin": None,
        "expected_log_growth": None,
    }

    if stage2_summary:
        method_payload = (stage2_summary.get("portfolio_methods") or {}).get(chosen_method)
        if isinstance(method_payload, dict):
            holdout = method_payload.get("holdout") or {}
            key_metrics["pf"] = _to_float(holdout.get("profit_factor"))
            key_metrics["maxdd"] = _to_float(holdout.get("max_drawdown"))

    if stage3_summary:
        overall = stage3_summary.get("overall_choice") or {}
        chosen_row = overall.get("chosen_row") or {}
        key_metrics["p_ruin"] = _to_float(chosen_row.get("p_ruin"))
        key_metrics["expected_log_growth"] = _to_float(chosen_row.get("expected_log_growth"))

    charts_index = find_frontier_selector_tables(pipeline_run_dir)
    _write_json_atomic(out_dir / "charts_index.json", charts_index)

    report_files = [str(path.resolve()) for path in find_best_report_files(pipeline_run_dir)]
    _write_json_atomic(out_dir / "reports_index.json", {"reports": report_files})

    window_start_ts, window_end_ts = _copy_equity_and_exposure(
        out_dir=out_dir,
        stage2_run_id=stage2_run_id,
        chosen_method=chosen_method,
        warnings=warnings,
    )
    _copy_trades_and_events(
        out_dir=out_dir,
        stage4_run_id=stage4_run_id,
        stage4_sim_run_id=stage4_sim_run_id,
        warnings=warnings,
    )

    config_hash = str(pipeline_summary.get("config_hash") or (compute_config_hash(pipeline_cfg) if pipeline_cfg else ""))
    data_hash = str(pipeline_summary.get("data_hash") or "")

    summary_ui = {
        "run_id": run_id,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stages": {
            "stage1_run_id": stage1_run_id,
            "stage2_run_id": stage2_run_id,
            "stage3_2_run_id": stage3_2_run_id,
            "stage3_3_run_id": stage3_3_run_id,
            "stage4_run_id": stage4_run_id,
            "stage4_sim_run_id": stage4_sim_run_id,
        },
        "symbols": symbols,
        "timeframe": timeframe,
        "evaluation_window_months": evaluation_window_months,
        "chosen_method": chosen_method,
        "chosen_leverage": chosen_leverage,
        "execution_mode": execution_mode,
        "key_metrics": key_metrics,
        "config_hash": config_hash,
        "data_hash": data_hash,
        "seed": seed,
        "resolved_end_ts": resolved_end_ts,
        "run_window_start_ts": window_start_ts,
        "run_window_end_ts": window_end_ts,
        "policy_snapshot": policy_snapshot,
        "bundle_build_warnings": warnings,
    }
    _write_json_atomic(out_dir / "summary_ui.json", summary_ui)


def build_ui_bundle_from_pipeline(run_dir: Path) -> None:
    """Build ui_bundle for a pipeline run directory."""

    run_dir = Path(run_dir)
    build_ui_bundle(
        run_id=run_dir.name,
        source_paths={
            "pipeline_run_dir": run_dir,
            "pipeline_summary_path": run_dir / "pipeline_summary.json",
            "pipeline_config_path": run_dir / "pipeline_config.yaml",
        },
        out_dir=run_dir / "ui_bundle",
    )


def _copy_equity_and_exposure(
    out_dir: Path,
    stage2_run_id: str,
    chosen_method: str,
    warnings: list[str],
) -> tuple[str | None, str | None]:
    if not stage2_run_id:
        warnings.append("stage2_run_id missing; equity/exposure unavailable")
        return None, None

    stage2_dir = RUNS_DIR / stage2_run_id
    name_map = {
        "equal": "portfolio_equal_weight.csv",
        "vol": "portfolio_vol_weight.csv",
        "corr-min": "portfolio_corr_min.csv",
        "corr": "portfolio_corr_min.csv",
        "min": "portfolio_corr_min.csv",
    }
    source_name = name_map.get(chosen_method, "portfolio_equal_weight.csv")
    source = stage2_dir / source_name
    if not source.exists():
        warnings.append(f"missing stage2 portfolio series: {source}")
        return None, None

    frame = _safe_csv(source)
    if frame.empty:
        warnings.append(f"empty portfolio series: {source}")
        return None, None

    if "timestamp" not in frame.columns or "equity" not in frame.columns:
        warnings.append(f"portfolio series missing required columns in {source}")
        return None, None

    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    normalized.to_csv(out_dir / "equity_curve.csv", index=False)

    roll_max = normalized["equity"].cummax().replace(0, pd.NA)
    drawdown = (roll_max - normalized["equity"]) / roll_max
    drawdown = drawdown.fillna(0.0)
    draw_frame = pd.DataFrame(
        {
            "timestamp": normalized["timestamp"],
            "drawdown": drawdown,
        }
    )
    draw_frame.to_csv(out_dir / "drawdown_curve.csv", index=False)

    if "exposure" in normalized.columns:
        exposure = normalized[["timestamp", "exposure"]].copy()
        exposure.to_csv(out_dir / "exposure.csv", index=False)

    if normalized.empty:
        return None, None
    start_ts = pd.to_datetime(normalized["timestamp"].iloc[0], utc=True, errors="coerce")
    end_ts = pd.to_datetime(normalized["timestamp"].iloc[-1], utc=True, errors="coerce")
    return (
        start_ts.isoformat() if not pd.isna(start_ts) else None,
        end_ts.isoformat() if not pd.isna(end_ts) else None,
    )


def _copy_trades_and_events(
    out_dir: Path,
    stage4_run_id: str,
    stage4_sim_run_id: str,
    warnings: list[str],
) -> None:
    stage_candidates = [stage4_sim_run_id, stage4_run_id]
    source_dir: Path | None = None
    for run_id in stage_candidates:
        if not run_id:
            continue
        candidate = RUNS_DIR / run_id
        if candidate.exists():
            source_dir = candidate
            break

    if source_dir is None:
        warnings.append("stage4 run artifacts not found; trades/events unavailable")
        return

    trades_source = source_dir / "trades.csv"
    if not trades_source.exists():
        orders = source_dir / "orders.csv"
        if orders.exists():
            trades_source = orders
            warnings.append("trades.csv missing; using orders.csv as trades fallback")

    if trades_source.exists():
        frame = _safe_csv(trades_source)
        if not frame.empty:
            frame.to_csv(out_dir / "trades.csv", index=False)
        else:
            warnings.append(f"trade source exists but empty: {trades_source}")
    else:
        warnings.append("no trades/orders artifact found")

    events_source = source_dir / "killswitch_events.csv"
    if events_source.exists():
        events = _safe_csv(events_source)
        if not events.empty:
            events.to_csv(out_dir / "events.csv", index=False)
        else:
            warnings.append(f"events source exists but empty: {events_source}")

    playback_source = source_dir / "playback_state.csv"
    if playback_source.exists():
        playback = _safe_csv(playback_source)
        if not playback.empty:
            playback.to_csv(out_dir / "playback_state.csv", index=False)
        else:
            warnings.append(f"playback source exists but empty: {playback_source}")


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_policy_snapshot(
    stage4_run_id: str,
    pipeline_cfg: dict[str, Any],
    chosen_method: str,
    chosen_leverage: float,
    warnings: list[str],
) -> dict[str, Any]:
    if stage4_run_id:
        snapshot_path = RUNS_DIR / stage4_run_id / "policy_snapshot.json"
        payload = _safe_json(snapshot_path)
        if payload:
            return payload
        warnings.append(f"policy snapshot missing for stage4_run_id: {stage4_run_id}")

    risk_cfg = pipeline_cfg.get("risk", {}) if isinstance(pipeline_cfg, dict) else {}
    costs_cfg = pipeline_cfg.get("costs", {}) if isinstance(pipeline_cfg, dict) else {}
    execution_cfg = pipeline_cfg.get("execution", {}) if isinstance(pipeline_cfg, dict) else {}
    return {
        "selected_method": str(chosen_method),
        "leverage": float(chosen_leverage),
        "execution_mode": str(execution_cfg.get("mode", "net")),
        "caps": {
            "max_gross_exposure": float(risk_cfg.get("max_gross_exposure", 0.0)),
            "max_net_exposure_per_symbol": float(risk_cfg.get("max_net_exposure_per_symbol", 0.0)),
            "max_open_positions": int(risk_cfg.get("max_open_positions", 0)),
        },
        "costs": {
            "round_trip_cost_pct": float(costs_cfg.get("round_trip_cost_pct", 0.0)),
            "slippage_pct": float(costs_cfg.get("slippage_pct", 0.0)),
            "funding_pct_per_day": float(costs_cfg.get("funding_pct_per_day", 0.0)),
        },
        "kill_switch": dict(risk_cfg.get("killswitch", {})),
        "source": "ui_bundle_fallback",
    }


def _validate_lineage_fields(status: str, stage1_run_id: str, stage2_run_id: str, stage3_3_run_id: str) -> None:
    if str(status).lower() != "success":
        return

    missing: list[str] = []
    if not str(stage1_run_id).strip():
        missing.append("stage1_run_id")
    if not str(stage2_run_id).strip():
        missing.append("stage2_run_id")
    if not str(stage3_3_run_id).strip():
        missing.append("stage3_3_run_id")

    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"ui_bundle lineage validation failed: missing required field(s): {joined}")


def _safe_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)
