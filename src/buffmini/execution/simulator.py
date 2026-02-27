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
    scaled_events = 0
    cooldown_bars = 0
    overlap_events = 0
    active_orders: list[Order] = []

    for bar_index, ts in enumerate(timestamps):
        symbol_exposure: dict[str, float] = {}
        for order in active_orders:
            symbol_exposure[order.symbol] = symbol_exposure.get(order.symbol, 0.0) + float(order.direction) * float(order.notional_fraction_of_equity) * float(order.leverage)

        forced_pnl_by_index = cfg.get("_forced_pnl_by_index", {})
        if isinstance(forced_pnl_by_index, dict) and int(bar_index) in forced_pnl_by_index:
            pnl_change = float(forced_pnl_by_index[int(bar_index)])
        else:
            pnl_change = 0.0
            for symbol in sorted(symbol_exposure.keys()):
                bar_return = float(rng.normal(0.0, 0.001))
                pnl_change += float(state.equity) * float(symbol_exposure[symbol]) * bar_return

        runtime_cfg = dict(cfg)
        runtime_cfg["_runtime_pnl_change"] = float(pnl_change)
        runtime_cfg["_runtime_bar_index"] = int(bar_index)
        prev_cooldown = int(state.cooldown_remaining_bars)
        orders = generate_orders(
            signals=grouped[ts],
            cfg=runtime_cfg,
            state=state,
            chosen_leverage=float(chosen_leverage),
            method_weights=cfg.get("_method_weights", {}),
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
                    }
                )
                for note in order.notes:
                    if note.startswith("cap_scale="):
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

        gross_exposure = float(sum(abs(float(item["direction"]) * float(item["notional_fraction_of_equity"]) * float(item["leverage"])) for item in state.open_positions))
        net_exposure = float(sum(float(item["direction"]) * float(item["notional_fraction_of_equity"]) * float(item["leverage"]) for item in state.open_positions))
        exposure_rows.append(
            {
                "ts": _ensure_utc(ts).isoformat(),
                "equity": float(state.equity),
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "open_positions": int(len(state.open_positions)),
                "cooldown_remaining_bars": int(state.cooldown_remaining_bars),
            }
        )

    exposure_df = pd.DataFrame(exposure_rows)
    orders_df = pd.DataFrame(order_rows)
    killswitch_df = pd.DataFrame(killswitch_rows)
    total_bars = len(timestamps)
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
        "seed": int(seed),
    }
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
    metrics, exposure_df, orders_df, killswitch_df = simulate_execution(
        signals_by_ts=grouped,
        cfg=runtime_cfg,
        initial_equity=float(cfg["portfolio"]["leverage_selector"]["initial_equity"]),
        chosen_leverage=float(chosen_leverage),
        seed=int(seed),
    )

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
    }
    run_id = f"{utc_now_compact()}_{stable_hash(payload, length=12)}_stage4_sim"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "execution_metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    exposure_df.to_csv(run_dir / "exposure_timeseries.csv", index=False)
    orders_df.to_csv(run_dir / "orders.csv", index=False)
    killswitch_df.to_csv(run_dir / "killswitch_events.csv", index=False)
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
