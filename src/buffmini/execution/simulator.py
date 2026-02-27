"""Offline Stage-4 paper-execution simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.execution.allocator import Order, generate_orders
from buffmini.execution.policy import Signal
from buffmini.execution.risk import PortfolioState
from buffmini.portfolio.monte_carlo import _load_stage2_context, _normalize_method_key
from buffmini.regime.classifier import REGIME_RANGE, regime_distribution_percent
from buffmini.risk.dynamic_leverage import compute_dynamic_leverage, compute_recent_drawdown
from buffmini.utils.hashing import stable_hash
from buffmini.utils.logging import get_logger
from buffmini.utils.time import utc_now_compact


logger = get_logger(__name__)


def build_signals_from_stage2(
    stage2_run_id: str,
    method: str,
    runs_dir: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
) -> tuple[list[Signal], dict[str, float], dict[str, Any]]:
    """Reconstruct Stage-2 candidate-entry signals for Stage-4 simulation."""

    method_key = _normalize_method_key(method)
    context = _load_stage2_context(stage2_run_id=stage2_run_id, runs_dir=runs_dir, data_dir=data_dir)
    payload = context.stage2_summary["portfolio_methods"].get(method_key)
    if payload is None:
        raise ValueError(f"Stage-2 method not found: {method_key}")
    weights = {str(candidate_id): float(weight) for candidate_id, weight in payload.get("weights", {}).items()}
    timeframe = str(context.config["universe"].get("timeframe", "1h"))
    candidate_metrics = _load_candidate_metrics(
        stage1_run_id=context.stage1_run_id,
        candidate_ids=set(weights.keys()),
        runs_dir=runs_dir,
    )
    regime_by_timestamp = _build_regime_lookup(context=context)

    signals: list[Signal] = []
    for candidate_id, bundle in context.candidate_bundles.items():
        weight = float(weights.get(candidate_id, 0.0))
        if weight <= 0:
            continue
        trades = bundle.get("trades", pd.DataFrame())
        if trades.empty:
            continue
        for row in trades.to_dict(orient="records"):
            entry_ts = pd.to_datetime(row.get("entry_time"), utc=True, errors="coerce")
            if pd.isna(entry_ts):
                continue
            side = str(row.get("side", "long")).lower()
            direction = 1 if side == "long" else -1
            entry_price = float(row.get("entry_price", 0.0) or 0.0)
            exit_price = float(row.get("exit_price", entry_price) or entry_price)
            stop_distance = abs((entry_price - exit_price) / entry_price) if entry_price > 0 else 0.01
            stop_distance = max(0.005, min(0.20, float(stop_distance)))
            signals.append(
                Signal(
                    ts=entry_ts,
                    symbol=str(row.get("symbol", "UNKNOWN")),
                    direction=direction,
                    strength=float(abs(weight)),
                    stop_distance=stop_distance,
                    strategy_id=str(candidate_id),
                    timeframe=timeframe,
                )
            )

    signals = sorted(signals, key=lambda item: (_ensure_utc(item.ts), item.symbol, item.strategy_id))
    metadata = {
        "stage1_run_id": context.stage1_run_id,
        "method": method_key,
        "stage2_summary": context.stage2_summary,
        "config_hash": context.config_hash,
        "data_hash": context.data_hash,
        "candidate_metrics": candidate_metrics,
        "regime_by_timestamp": regime_by_timestamp,
    }
    return signals, weights, metadata


def simulate_execution(
    signals_by_ts: dict[pd.Timestamp, list[Signal]] | list[Signal],
    cfg: dict[str, Any],
    initial_equity: float,
    chosen_leverage: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate Stage-4 execution decisions on historical signal stream."""

    grouped = _normalize_signals(signals_by_ts)
    timestamps = sorted(grouped.keys())
    stage6_cfg = (cfg.get("evaluation", {}) or {}).get("stage6", {})
    stage6_enabled = bool(stage6_cfg.get("enabled", False))
    dynamic_cfg = dict(stage6_cfg.get("dynamic_leverage", {})) if isinstance(stage6_cfg, dict) else {}
    if not dynamic_cfg.get("allowed_levels"):
        dynamic_cfg["allowed_levels"] = (
            cfg.get("portfolio", {})
            .get("leverage_selector", {})
            .get("leverage_levels", [])
        )
    dd_lookback_bars = int(dynamic_cfg.get("dd_lookback_bars", 168))
    regime_lookup = cfg.get("_runtime_regime_by_timestamp", {}) if isinstance(cfg.get("_runtime_regime_by_timestamp", {}), dict) else {}
    candidate_metrics = cfg.get("_runtime_candidate_metrics", {}) if isinstance(cfg.get("_runtime_candidate_metrics", {}), dict) else {}

    if not timestamps:
        empty = pd.DataFrame()
        metrics = {
            "order_count": 0,
            "total_bars": 0,
            "percent_time_in_cooldown": 0.0,
            "scaled_event_count": 0,
            "killswitch_event_count": 0,
            "opposite_signal_events": 0,
            "overlap_rate": 0.0,
            "gross_exposure_max": 0.0,
            "gross_exposure_mean": 0.0,
            "chosen_leverage": float(chosen_leverage),
            "stage6_enabled": bool(stage6_enabled),
            "base_leverage": float(chosen_leverage),
            "avg_leverage": float(chosen_leverage),
            "regime_distribution": {REGIME_RANGE: 100.0},
            "seed": int(seed),
        }
        return metrics, empty, empty, empty

    rng = np.random.default_rng(int(seed))
    state = PortfolioState(
        equity=float(initial_equity),
        peak_equity=float(initial_equity),
        day_start_equity=float(initial_equity),
        open_positions=[],
    )

    exposure_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    killswitch_rows: list[dict[str, Any]] = []
    sizing_rows: list[dict[str, Any]] = []
    leverage_rows: list[dict[str, Any]] = []
    scaled_events = 0
    cooldown_bars = 0
    overlap_events = 0
    active_orders: list[Order] = []
    equity_history: list[float] = [float(initial_equity)]

    for bar_index, ts in enumerate(timestamps):
        symbol_exposure: dict[str, float] = {}
        for order in active_orders:
            symbol_exposure[order.symbol] = (
                symbol_exposure.get(order.symbol, 0.0)
                + float(order.direction) * float(order.notional_fraction_of_equity) * float(order.leverage)
            )

        forced_pnl_by_index = cfg.get("_forced_pnl_by_index", {})
        if isinstance(forced_pnl_by_index, dict) and int(bar_index) in forced_pnl_by_index:
            pnl_change = float(forced_pnl_by_index[int(bar_index)])
        else:
            pnl_change = 0.0
            for symbol in sorted(symbol_exposure.keys()):
                bar_return = float(rng.normal(0.0, 0.001))
                pnl_change += float(state.equity) * float(symbol_exposure[symbol]) * bar_return

        regime = _resolve_regime_for_timestamp(ts, regime_lookup)
        dd_recent = compute_recent_drawdown(equity_history=equity_history, lookback_bars=dd_lookback_bars)
        leverage_info = {
            "regime": regime,
            "dd_recent": float(dd_recent),
            "leverage_raw": float(chosen_leverage),
            "leverage_clipped": float(chosen_leverage),
        }
        if stage6_enabled:
            leverage_info = compute_dynamic_leverage(
                base_leverage=float(chosen_leverage),
                regime=regime,
                dd_recent=float(dd_recent),
                config=dynamic_cfg,
            )
        current_leverage = float(leverage_info["leverage_clipped"])

        runtime_cfg = dict(cfg)
        runtime_cfg["_runtime_pnl_change"] = float(pnl_change)
        runtime_cfg["_runtime_bar_index"] = int(bar_index)
        runtime_cfg["_runtime_leverage"] = float(current_leverage)
        runtime_cfg["_runtime_regime"] = str(regime)
        runtime_cfg["_runtime_candidate_metrics"] = candidate_metrics

        prev_cooldown = int(state.cooldown_remaining_bars)
        orders = generate_orders(
            signals=grouped[ts],
            cfg=runtime_cfg,
            state=state,
            chosen_leverage=float(chosen_leverage),
            method_weights=cfg.get("_method_weights", {}),
        )

        for row in runtime_cfg.get("_last_sizing_records", []) or []:
            sizing_rows.append(
                {
                    "timestamp": _ensure_utc(row.get("timestamp", ts)).isoformat(),
                    "component_id": str(row.get("component_id", "")),
                    "symbol": str(row.get("symbol", "")),
                    "confidence": float(row.get("confidence", 0.0)),
                    "multiplier": float(row.get("multiplier", 1.0)),
                    "applied_weight": float(row.get("applied_weight", 0.0)),
                    "base_weight": float(row.get("base_weight", 0.0)),
                    "component_renorm_scale": float(row.get("component_renorm_scale", 1.0)),
                    "cap_scale": float(row.get("cap_scale", 1.0)),
                    "final_net_exposure": float(row.get("final_net_exposure", 0.0)),
                    "regime": str(regime),
                    "leverage": float(current_leverage),
                }
            )

        decision = runtime_cfg.get("_last_risk_decision")
        if decision is not None and not bool(decision.allow_new_trades):
            cooldown_bars += 1
        if prev_cooldown == 0 and int(state.cooldown_remaining_bars) > 0:
            killswitch_rows.append(
                {
                    "ts": _ensure_utc(ts).isoformat(),
                    "reason": ",".join(getattr(decision, "reasons", []) or ["threshold_triggered"]),
                    "cool_down_bars": int(cfg["risk"]["killswitch"]["cool_down_bars"]),
                    "equity": float(state.equity),
                }
            )

        if orders:
            active_orders = orders
            state.open_positions = [
                {
                    "symbol": order.symbol,
                    "direction": int(order.direction),
                    "notional_fraction_of_equity": float(order.notional_fraction_of_equity),
                    "leverage": float(order.leverage),
                    "strategy_id": order.strategy_id,
                    "regime": str(regime),
                }
                for order in orders
            ]
            for order in orders:
                order_rows.append(
                    {
                        "ts": _ensure_utc(order.ts).isoformat(),
                        "symbol": order.symbol,
                        "direction": int(order.direction),
                        "notional_fraction_of_equity": float(order.notional_fraction_of_equity),
                        "leverage": float(order.leverage),
                        "strategy_id": order.strategy_id,
                        "notes": ";".join(order.notes),
                        "regime": str(regime),
                    }
                )
                for note in order.notes:
                    if note.startswith("cap_scale=") or note.startswith("component_scale="):
                        try:
                            if float(note.split("=", 1)[1]) < 0.999999:
                                scaled_events += 1
                        except ValueError:
                            pass

        directions_by_symbol: dict[str, set[int]] = {}
        for signal in grouped[ts]:
            directions_by_symbol.setdefault(signal.symbol, set()).add(1 if int(signal.direction) >= 0 else -1)
        if any(len(direction_set) > 1 for direction_set in directions_by_symbol.values()):
            overlap_events += 1

        gross_exposure = float(
            sum(
                abs(float(item["direction"]) * float(item["notional_fraction_of_equity"]) * float(item["leverage"]))
                for item in state.open_positions
            )
        )
        net_exposure = float(
            sum(
                float(item["direction"]) * float(item["notional_fraction_of_equity"]) * float(item["leverage"])
                for item in state.open_positions
            )
        )
        exposure_rows.append(
            {
                "ts": _ensure_utc(ts).isoformat(),
                "equity": float(state.equity),
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "open_positions": int(len(state.open_positions)),
                "cooldown_remaining_bars": int(state.cooldown_remaining_bars),
                "regime": str(regime),
                "leverage": float(current_leverage),
            }
        )
        leverage_rows.append(
            {
                "timestamp": _ensure_utc(ts).isoformat(),
                "regime": str(regime),
                "leverage_raw": float(leverage_info["leverage_raw"]),
                "leverage_clipped": float(leverage_info["leverage_clipped"]),
                "dd_recent": float(leverage_info["dd_recent"]),
            }
        )
        equity_history.append(float(state.equity))

    exposure_df = pd.DataFrame(exposure_rows)
    orders_df = pd.DataFrame(order_rows)
    killswitch_df = pd.DataFrame(killswitch_rows)
    sizing_df = pd.DataFrame(sizing_rows)
    leverage_df = pd.DataFrame(leverage_rows)
    total_bars = len(timestamps)
    avg_leverage = float(leverage_df["leverage_clipped"].mean()) if not leverage_df.empty else float(chosen_leverage)
    regime_dist = regime_distribution_percent(leverage_df, column="regime")
    metrics = {
        "order_count": int(len(orders_df)),
        "total_bars": int(total_bars),
        "percent_time_in_cooldown": float(cooldown_bars / total_bars) if total_bars > 0 else 0.0,
        "scaled_event_count": int(scaled_events),
        "killswitch_event_count": int(len(killswitch_df)),
        "opposite_signal_events": int(overlap_events),
        "overlap_rate": float(overlap_events / total_bars) if total_bars > 0 else 0.0,
        "gross_exposure_max": float(exposure_df["gross_exposure"].max()) if not exposure_df.empty else 0.0,
        "gross_exposure_mean": float(exposure_df["gross_exposure"].mean()) if not exposure_df.empty else 0.0,
        "chosen_leverage": float(chosen_leverage),
        "stage6_enabled": bool(stage6_enabled),
        "base_leverage": float(chosen_leverage),
        "avg_leverage": float(avg_leverage),
        "regime_distribution": regime_dist,
        "seed": int(seed),
    }
    if not sizing_df.empty:
        cfg["_stage6_sizing_df"] = sizing_df
    if not leverage_df.empty:
        cfg["_stage6_leverage_df"] = leverage_df
    return metrics, exposure_df, orders_df, killswitch_df


def run_stage4_simulation(
    stage2_run_id: str,
    cfg: dict[str, Any],
    stage3_choice: dict[str, Any] | None,
    days: int | None = 90,
    bars: int | None = None,
    runs_dir: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
    seed: int = 42,
) -> Path:
    """High-level Stage-4 simulation runner used by CLI/UI."""

    selected_method, chosen_leverage, _, _ = resolve_stage4_method_and_leverage(cfg=cfg, stage3_choice=stage3_choice)
    signals, method_weights, metadata = build_signals_from_stage2(
        stage2_run_id=stage2_run_id,
        method=selected_method,
        runs_dir=runs_dir,
        data_dir=data_dir,
    )

    if int(bars or 0) > 0:
        signals = signals[-int(bars) :]
    elif int(days or 0) > 0:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(days))
        signals = [signal for signal in signals if _ensure_utc(signal.ts) >= cutoff]

    grouped = _normalize_signals(signals)
    runtime_cfg = dict(cfg)
    runtime_cfg["_method_weights"] = method_weights
    runtime_cfg["_runtime_candidate_metrics"] = metadata.get("candidate_metrics", {})
    runtime_cfg["_runtime_regime_by_timestamp"] = metadata.get("regime_by_timestamp", {})
    metrics, exposure_df, orders_df, killswitch_df = simulate_execution(
        signals_by_ts=grouped,
        cfg=runtime_cfg,
        initial_equity=float(cfg["portfolio"]["leverage_selector"]["initial_equity"]),
        chosen_leverage=float(chosen_leverage),
        seed=int(seed),
    )
    sizing_df = runtime_cfg.get("_stage6_sizing_df", pd.DataFrame())
    leverage_df = runtime_cfg.get("_stage6_leverage_df", pd.DataFrame())

    payload = {
        "stage2_run_id": stage2_run_id,
        "stage1_run_id": metadata["stage1_run_id"],
        "method": selected_method,
        "chosen_leverage": float(chosen_leverage),
        "seed": int(seed),
        "days": int(days) if days is not None else None,
        "bars": int(bars) if bars is not None else None,
        "metrics": metrics,
        "config_hash": metadata["config_hash"],
        "data_hash": metadata["data_hash"],
        "stage6_enabled": bool((cfg.get("evaluation", {}) or {}).get("stage6", {}).get("enabled", False)),
        "base_leverage": float(chosen_leverage),
        "avg_leverage": float(metrics.get("avg_leverage", chosen_leverage)),
        "regime_distribution": metrics.get("regime_distribution", {}),
    }
    run_id = f"{utc_now_compact()}_{stable_hash(payload, length=12)}_stage4_sim"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "execution_metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    exposure_df.to_csv(run_dir / "exposure_timeseries.csv", index=False)
    orders_df.to_csv(run_dir / "orders.csv", index=False)
    trades_df = _orders_to_trade_events(orders_df)
    trades_df.to_csv(run_dir / "trades.csv", index=False)
    killswitch_df.to_csv(run_dir / "killswitch_events.csv", index=False)
    if isinstance(sizing_df, pd.DataFrame) and not sizing_df.empty:
        sizing_df.to_csv(run_dir / "sizing_multipliers.csv", index=False)
    if isinstance(leverage_df, pd.DataFrame) and not leverage_df.empty:
        leverage_df.to_csv(run_dir / "leverage_path.csv", index=False)
    playback_df = _build_playback_state(exposure_df=exposure_df, orders_df=orders_df, killswitch_df=killswitch_df)
    playback_df.to_csv(run_dir / "playback_state.csv", index=False)
    logger.info("Saved Stage-4 simulation artifacts to %s", run_dir)
    return run_dir


def resolve_stage4_method_and_leverage(
    cfg: dict[str, Any],
    stage3_choice: dict[str, Any] | None,
) -> tuple[str, float, bool, list[str]]:
    """Resolve method/leverage source with optional Stage-3.3 override."""

    warnings: list[str] = []
    stage4_cfg = cfg["evaluation"]["stage4"]
    selected_method = str(stage4_cfg["default_method"])
    selected_leverage = float(stage4_cfg["default_leverage"])
    from_stage3 = False
    if stage3_choice is not None:
        overall = stage3_choice.get("overall_choice", {})
        if overall.get("status") == "OK" and overall.get("method") is not None and overall.get("chosen_leverage") is not None:
            selected_method = str(overall["method"])
            selected_leverage = float(overall["chosen_leverage"])
            from_stage3 = True
        else:
            warnings.append("Stage-3.3 choice present but no feasible overall selection; fallback defaults used.")
    return selected_method, selected_leverage, from_stage3, warnings


def _load_candidate_metrics(
    stage1_run_id: str,
    candidate_ids: set[str],
    runs_dir: Path,
) -> dict[str, dict[str, float]]:
    if not stage1_run_id or not candidate_ids:
        return {}
    candidates_dir = Path(runs_dir) / stage1_run_id / "candidates"
    if not candidates_dir.exists():
        return {}
    metrics: dict[str, dict[str, float]] = {}
    for path in sorted(candidates_dir.glob("strategy_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidate_id = str(payload.get("candidate_id", ""))
        if candidate_id not in candidate_ids:
            continue
        metrics[candidate_id] = {
            "exp_lcb_holdout": _to_float(payload.get("exp_lcb_holdout"), 0.0),
            "effective_edge": _to_float(payload.get("effective_edge"), 0.0),
            "pf_adj_holdout": _to_float(payload.get("pf_adj_holdout"), 1.0),
            "trades_per_month_holdout": _to_float(payload.get("trades_per_month_holdout"), 0.0),
        }
    return metrics


def _build_regime_lookup(context: Any) -> dict[str, str]:
    regime_by_ts: dict[str, str] = {}
    if not getattr(context, "signal_caches", None):
        return regime_by_ts
    # Feature values are identical across candidate caches; use first available cache.
    first_cache = next(iter(context.signal_caches.values()), {})
    if not isinstance(first_cache, dict):
        return regime_by_ts
    for frame in first_cache.values():
        if frame is None or frame.empty or "timestamp" not in frame.columns:
            continue
        if "regime" not in frame.columns:
            continue
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        regimes = frame["regime"].astype(str)
        for ts, regime in zip(timestamps, regimes, strict=False):
            if pd.isna(ts):
                continue
            regime_by_ts[pd.Timestamp(ts).isoformat()] = str(regime)
    return regime_by_ts


def _resolve_regime_for_timestamp(ts: pd.Timestamp, regime_lookup: dict[str, str]) -> str:
    key = _ensure_utc(ts).isoformat()
    return str(regime_lookup.get(key, REGIME_RANGE))


def _normalize_signals(signals_by_ts: dict[pd.Timestamp, list[Signal]] | list[Signal]) -> dict[pd.Timestamp, list[Signal]]:
    if isinstance(signals_by_ts, dict):
        grouped: dict[pd.Timestamp, list[Signal]] = {}
        for key, bucket in signals_by_ts.items():
            ts = pd.Timestamp(key)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            grouped[ts] = list(bucket)
        return dict(sorted(grouped.items(), key=lambda item: item[0]))

    grouped: dict[pd.Timestamp, list[Signal]] = {}
    for signal in signals_by_ts:
        ts = pd.Timestamp(signal.ts)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        grouped.setdefault(ts, []).append(signal)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _ensure_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _orders_to_trade_events(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Convert order stream to lightweight trade-event table for UI plotting."""

    if orders_df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "direction",
                "action",
                "strategy_id",
                "notional_fraction_of_equity",
                "leverage",
                "notes",
            ]
        )
    frame = orders_df.copy()
    frame = frame.rename(columns={"ts": "timestamp"})
    frame["action"] = "entry"
    if "notes" not in frame.columns:
        frame["notes"] = ""
    columns = [
        "timestamp",
        "symbol",
        "direction",
        "action",
        "strategy_id",
        "notional_fraction_of_equity",
        "leverage",
        "notes",
    ]
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns].copy()


def _build_playback_state(
    exposure_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    killswitch_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-bar playback contract for Stage-5 paper trading UI."""

    if exposure_df.empty:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "action", "exposure", "reason", "equity"]
        )

    exposure = exposure_df.copy()
    exposure["timestamp"] = pd.to_datetime(exposure["ts"], utc=True, errors="coerce")
    exposure = exposure.dropna(subset=["timestamp"]).sort_values("timestamp")

    rows: list[dict[str, Any]] = []
    order_frame = orders_df.copy() if not orders_df.empty else pd.DataFrame()
    if not order_frame.empty:
        order_frame["timestamp"] = pd.to_datetime(order_frame["ts"], utc=True, errors="coerce")

    kill_frame = killswitch_df.copy() if not killswitch_df.empty else pd.DataFrame()
    if not kill_frame.empty:
        kill_frame["timestamp"] = pd.to_datetime(kill_frame["ts"], utc=True, errors="coerce")

    for row in exposure.to_dict(orient="records"):
        ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "symbol": "ALL",
                "action": "hold",
                "exposure": float(row.get("gross_exposure", 0.0) or 0.0),
                "reason": "",
                "equity": float(row.get("equity", 0.0) or 0.0),
            }
        )

        if not order_frame.empty:
            orders_now = order_frame[order_frame["timestamp"] == ts]
            for order in orders_now.to_dict(orient="records"):
                rows.append(
                    {
                        "timestamp": ts.isoformat(),
                        "symbol": str(order.get("symbol", "UNKNOWN")),
                        "action": "open",
                        "exposure": float(order.get("notional_fraction_of_equity", 0.0) or 0.0),
                        "reason": str(order.get("notes", "")),
                        "equity": float(row.get("equity", 0.0) or 0.0),
                    }
                )

        if not kill_frame.empty:
            kills_now = kill_frame[kill_frame["timestamp"] == ts]
            for event in kills_now.to_dict(orient="records"):
                rows.append(
                    {
                        "timestamp": ts.isoformat(),
                        "symbol": "ALL",
                        "action": "close",
                        "exposure": float(row.get("gross_exposure", 0.0) or 0.0),
                        "reason": str(event.get("reason", "killswitch")),
                        "equity": float(row.get("equity", 0.0) or 0.0),
                    }
                )

    playback = pd.DataFrame(rows)
    if playback.empty:
        return playback
    playback = playback.sort_values(["timestamp", "symbol", "action"]).reset_index(drop=True)
    return playback
