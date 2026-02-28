"""Generate Stage-9.3 recent OI overlay evidence report from local data."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.analysis.impact_analysis import bootstrap_median_difference, compute_forward_returns
from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, trend_pullback
from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR
from buffmini.data.derived_store import load_derived_parquet
from buffmini.data.features import calculate_features
from buffmini.data.store import build_data_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-9.3 recent OI overlay evidence report")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out-md", type=Path, default=Path("docs") / "stage9_3_recent_oi_overlay.md")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("docs") / "stage9_3_recent_oi_overlay_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    config = load_config(args.config)
    cfg = deepcopy(config)
    cfg.setdefault("data", {})
    cfg["data"]["include_futures_extras"] = True

    extras_cfg = cfg["data"].get("futures_extras", {})
    symbols = list(extras_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"]))
    timeframe = str(extras_cfg.get("timeframe", cfg.get("universe", {}).get("timeframe", "1h")))
    if timeframe != "1h":
        raise ValueError("Stage-9.3 overlay report only supports 1h timeframe")

    stage93 = cfg.get("evaluation", {}).get("stage9_3", {})
    report_windows = [int(value) for value in stage93.get("report_windows_days", [30, 60, 90])]
    report_windows = sorted({int(value) for value in report_windows if int(value) > 0})
    if not report_windows:
        raise ValueError("evaluation.stage9_3.report_windows_days must be non-empty")

    store = build_data_store(
        backend=str(cfg.get("data", {}).get("backend", "parquet")),
        data_dir=args.data_dir,
        base_timeframe=str(cfg.get("universe", {}).get("base_timeframe") or timeframe),
        resample_source=str(cfg.get("data", {}).get("resample_source", "direct")),
        derived_dir=args.derived_dir,
        partial_last_bucket=bool(cfg.get("data", {}).get("partial_last_bucket", False)),
    )

    summary_symbols: dict[str, Any] = {}
    all_effect_rows: list[dict[str, Any]] = []

    for symbol in symbols:
        frame = store.load_ohlcv(symbol=symbol, timeframe=timeframe)
        if frame.empty:
            raise RuntimeError(f"No OHLCV data found for {symbol} {timeframe}")

        funding = load_derived_parquet(
            kind="funding",
            symbol=symbol,
            timeframe=timeframe,
            data_dir=args.derived_dir,
        )
        oi = load_derived_parquet(
            kind="open_interest",
            symbol=symbol,
            timeframe=timeframe,
            data_dir=args.derived_dir,
        )

        oi_ts = _extract_oi_ts(oi)
        oi_total_rows = int(len(oi))
        oi_availability = {
            "earliest_oi_ts": oi_ts.iloc[0].isoformat() if not oi_ts.empty else None,
            "latest_oi_ts": oi_ts.iloc[-1].isoformat() if not oi_ts.empty else None,
            "row_count": int(len(oi_ts)),
            "raw_rows": oi_total_rows,
        }

        symbol_windows: list[dict[str, Any]] = []
        for requested_days in report_windows:
            window_cfg = _overlay_cfg(cfg=cfg, recent_window_days=int(requested_days))
            features = calculate_features(
                frame,
                config=window_cfg,
                symbol=symbol,
                timeframe=timeframe,
                derived_data_dir=args.derived_dir,
            )
            overlay = dict(features.attrs.get("oi_overlay", {}) or {})
            oi_active = pd.Series(features.get("oi_active", False), index=features.index).fillna(False).astype(bool)
            active_pct = float(oi_active.mean() * 100.0) if len(oi_active) > 0 else 0.0
            effects = _recent_window_effects(
                features=features,
                oi_active=oi_active,
                seed=int(args.seed) + int(requested_days),
                n_boot=int(args.n_boot),
            )
            for row in effects:
                enriched = dict(row)
                enriched["symbol"] = symbol
                enriched["requested_days"] = int(requested_days)
                all_effect_rows.append(enriched)

            symbol_windows.append(
                {
                    "requested_days": int(requested_days),
                    "clamped_days": int(overlay.get("oi_clamped_days", 0) or 0),
                    "window_start_ts": overlay.get("oi_window_start_ts"),
                    "window_end_ts": overlay.get("oi_window_end_ts"),
                    "window_note": str(overlay.get("oi_window_note", "")),
                    "percent_bars_oi_active": float(active_pct),
                    "oi_active_rows": int(oi_active.sum()),
                    "total_rows": int(len(features)),
                    "effects": effects,
                }
            )

        ab_result = _ab_non_corruption(
            frame=frame,
            base_cfg=cfg,
            symbol=symbol,
            timeframe=timeframe,
            derived_data_dir=args.derived_dir,
            recent_window_days=int(report_windows[0]),
        )

        summary_symbols[symbol] = {
            "oi_availability": oi_availability,
            "windows": symbol_windows,
            "ab_non_corruption": ab_result,
        }

    top_effects = sorted(all_effect_rows, key=lambda item: abs(float(item["median_diff"])), reverse=True)[:6]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "symbols": summary_symbols,
        "report_windows_days": report_windows,
        "nan_policy": str(stage93.get("nan_policy", "condition_false")),
        "statement": (
            "OI is unavailable outside the recent overlay window; rows outside oi_active are excluded "
            "from OI impact confirmation."
        ),
        "top_recent_effects": top_effects,
        "runtime_seconds": round(time.time() - started, 3),
    }

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(path=args.out_md, summary=summary)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote: {args.out_md}")
    print(f"wrote: {args.out_json}")


def _overlay_cfg(cfg: dict[str, Any], recent_window_days: int) -> dict[str, Any]:
    updated = deepcopy(cfg)
    updated.setdefault("data", {})
    updated["data"]["include_futures_extras"] = True
    extras = updated["data"].setdefault("futures_extras", {})
    oi_cfg = extras.setdefault("open_interest", {})
    overlay = oi_cfg.setdefault("overlay", {})
    overlay["enabled"] = True
    overlay["recent_window_days"] = int(recent_window_days)
    overlay["clamp_to_available"] = True
    overlay["inactive_value"] = "nan"
    return updated


def _extract_oi_ts(oi: pd.DataFrame) -> pd.Series:
    if oi.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    ts_col = "ts" if "ts" in oi.columns else "timestamp" if "timestamp" in oi.columns else None
    if ts_col is None:
        return pd.Series(dtype="datetime64[ns, UTC]")
    ts = pd.to_datetime(oi[ts_col], utc=True, errors="coerce")
    if "open_interest" in oi.columns:
        oi_values = pd.to_numeric(oi["open_interest"], errors="coerce")
        ts = ts[oi_values.notna()]
    ts = ts.dropna().sort_values().reset_index(drop=True)
    return ts


def _recent_window_effects(
    features: pd.DataFrame,
    oi_active: pd.Series,
    seed: int,
    n_boot: int,
) -> list[dict[str, Any]]:
    prepared = compute_forward_returns(features, horizons=(24, 72))
    subset = prepared.loc[oi_active].copy()
    if subset.empty:
        return []

    rows: list[dict[str, Any]] = []
    for condition in ("crowd_long_risk", "crowd_short_risk"):
        if condition not in subset.columns:
            continue
        for horizon in (24, 72):
            ret_col = f"forward_return_{horizon}h"
            if ret_col not in subset.columns:
                continue

            cond_mask = pd.to_numeric(subset[condition], errors="coerce").fillna(0) > 0
            values = pd.to_numeric(subset[ret_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            cond_values = values.loc[cond_mask].dropna()
            base_values = values.loc[~cond_mask].dropna()

            if cond_values.empty or base_values.empty:
                row = {
                    "condition": condition,
                    "horizon": f"{horizon}h",
                    "count": int(cond_values.size),
                    "median_diff": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                }
                rows.append(row)
                continue

            work = subset[[condition, ret_col]].copy()
            boot = bootstrap_median_difference(
                frame=work,
                condition_col=condition,
                value_col=ret_col,
                n_boot=int(n_boot),
                seed=int(seed) + int(horizon),
            )
            row = {
                "condition": condition,
                "horizon": f"{horizon}h",
                "count": int(cond_values.size),
                "median_diff": float(boot["median_diff"]),
                "ci_low": float(boot["ci_low"]),
                "ci_high": float(boot["ci_high"]),
            }
            rows.append(row)

    return rows


def _ab_non_corruption(
    frame: pd.DataFrame,
    base_cfg: dict[str, Any],
    symbol: str,
    timeframe: str,
    derived_data_dir: Path,
    recent_window_days: int,
) -> dict[str, Any]:
    cfg_a = deepcopy(base_cfg)
    cfg_a.setdefault("data", {})
    cfg_a["data"]["include_futures_extras"] = False

    cfg_b = _overlay_cfg(cfg=base_cfg, recent_window_days=int(recent_window_days))

    feature_a = calculate_features(
        frame,
        config=cfg_a,
        symbol=symbol,
        timeframe=timeframe,
        derived_data_dir=derived_data_dir,
    )
    feature_b = calculate_features(
        frame,
        config=cfg_b,
        symbol=symbol,
        timeframe=timeframe,
        derived_data_dir=derived_data_dir,
    )

    strategy = trend_pullback()
    bt_a = feature_a.copy()
    bt_a["signal"] = generate_signals(bt_a, strategy=strategy, gating_mode="none")
    bt_b = feature_b.copy()
    bt_b["signal"] = generate_signals(bt_b, strategy=strategy, gating_mode="none")

    costs = dict(base_cfg.get("costs", {}) or {})
    risk_cfg = dict(base_cfg.get("risk", {}) or {})
    round_trip_cost_pct = float(costs.get("round_trip_cost_pct", 0.1))
    slippage_pct = float(costs.get("slippage_pct", 0.0005))
    max_hold_bars = int(risk_cfg.get("max_holding_bars", 24))

    result_a = run_backtest(
        frame=bt_a,
        strategy_name=strategy.name,
        symbol=symbol,
        max_hold_bars=max_hold_bars,
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
    )
    result_b = run_backtest(
        frame=bt_b,
        strategy_name=strategy.name,
        symbol=symbol,
        max_hold_bars=max_hold_bars,
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
    )

    trades_a = float(result_a.metrics.get("trade_count", 0.0))
    trades_b = float(result_b.metrics.get("trade_count", 0.0))
    if trades_a <= 0:
        trade_count_delta_pct = 0.0
    else:
        trade_count_delta_pct = abs(trades_b - trades_a) / trades_a * 100.0

    equity_a = result_a.equity_curve["equity"].to_numpy(dtype=float)
    equity_b = result_b.equity_curve["equity"].to_numpy(dtype=float)
    same_shape = equity_a.shape == equity_b.shape
    equity_identical = bool(same_shape and np.allclose(equity_a, equity_b, atol=1e-12, rtol=0.0))

    return {
        "strategy": strategy.name,
        "trade_count_a": int(trades_a),
        "trade_count_b": int(trades_b),
        "trade_count_delta_pct": float(trade_count_delta_pct),
        "equity_curve_identical": equity_identical,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Stage-9.3 Recent OI Overlay")
    lines.append("")
    lines.append(f"- generated_at_utc: `{summary['generated_at_utc']}`")
    lines.append(f"- report_windows_days: `{summary['report_windows_days']}`")
    lines.append(f"- nan_policy: `{summary['nan_policy']}`")
    lines.append("")
    lines.append("## OI Availability Facts")
    lines.append("")

    for symbol, payload in summary["symbols"].items():
        availability = payload["oi_availability"]
        lines.append(f"### {symbol}")
        lines.append(
            "- "
            f"earliest_oi_ts=`{availability['earliest_oi_ts']}`, "
            f"latest_oi_ts=`{availability['latest_oi_ts']}`, "
            f"row_count=`{availability['row_count']}`, "
            f"raw_rows=`{availability['raw_rows']}`"
        )
        lines.append("")
        lines.append("| requested_days | clamped_days | start_ts | end_ts | note | oi_active_% | oi_active_rows | total_rows |")
        lines.append("| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |")
        for row in payload["windows"]:
            lines.append(
                f"| {int(row['requested_days'])} | {int(row['clamped_days'])} | "
                f"{row['window_start_ts']} | {row['window_end_ts']} | {row['window_note']} | "
                f"{float(row['percent_bars_oi_active']):.4f} | {int(row['oi_active_rows'])} | {int(row['total_rows'])} |"
            )
        lines.append("")

        lines.append("#### A/B Non-Corruption (Non-OI Baseline)")
        ab = payload["ab_non_corruption"]
        lines.append(
            "- "
            f"strategy=`{ab['strategy']}`, trade_count_delta_pct=`{float(ab['trade_count_delta_pct']):.6f}`, "
            f"equity_curve_identical=`{ab['equity_curve_identical']}`"
        )
        lines.append("")

        lines.append("#### Recent-Window Effects (oi_active only)")
        lines.append("| requested_days | condition | horizon | count | median_diff | ci_low | ci_high |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: |")
        for window in payload["windows"]:
            effects = list(window.get("effects", []))
            if not effects:
                lines.append(f"| {int(window['requested_days'])} | no_signal | 24h | 0 | 0.000000 | 0.000000 | 0.000000 |")
                continue
            for effect in effects:
                lines.append(
                    f"| {int(window['requested_days'])} | {effect['condition']} | {effect['horizon']} | "
                    f"{int(effect['count'])} | {float(effect['median_diff']):.6f} | "
                    f"{float(effect['ci_low']):.6f} | {float(effect['ci_high']):.6f} |"
                )
        lines.append("")

    lines.append("## Top Recent-Window Effects")
    if summary["top_recent_effects"]:
        lines.append("| symbol | requested_days | condition | horizon | count | median_diff | ci_low | ci_high |")
        lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |")
        for row in summary["top_recent_effects"]:
            lines.append(
                f"| {row['symbol']} | {int(row['requested_days'])} | {row['condition']} | {row['horizon']} | "
                f"{int(row['count'])} | {float(row['median_diff']):.6f} | "
                f"{float(row['ci_low']):.6f} | {float(row['ci_high']):.6f} |"
            )
    else:
        lines.append("- no recent-window effects available")
    lines.append("")

    lines.append("## Scope Statement")
    lines.append(f"- {summary['statement']}")
    lines.append("")
    lines.append(f"- runtime_seconds: `{summary['runtime_seconds']}`")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
