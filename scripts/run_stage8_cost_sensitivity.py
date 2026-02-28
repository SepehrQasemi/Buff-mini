"""Stage-8.2 offline cost sensitivity sweep (simple vs v2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from buffmini.backtest.engine import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-8 cost model sensitivity sweep")
    parser.add_argument("--output-md", type=Path, default=Path("docs/stage8_cost_sensitivity.md"))
    parser.add_argument("--output-json", type=Path, default=Path("docs/stage8_cost_sensitivity.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = _synthetic_frame()
    rows: list[dict[str, float | str]] = []

    scenarios = [
        {
            "label": "simple_delay0",
            "mode": "simple",
            "cost_model_cfg": {"mode": "simple", "round_trip_cost_pct": 0.1},
            "slippage_pct": 0.0005,
        },
        {
            "label": "v2_low_delay0",
            "mode": "v2",
            "cost_model_cfg": {
                "mode": "v2",
                "round_trip_cost_pct": 0.1,
                "v2": {
                    "slippage_bps_base": 0.5,
                    "slippage_bps_vol_mult": 1.5,
                    "spread_bps": 0.5,
                    "delay_bars": 0,
                    "vol_proxy": "atr_pct",
                    "vol_lookback": 14,
                    "max_total_bps_per_side": 10.0,
                },
            },
            "slippage_pct": 0.0005,
        },
        {
            "label": "v2_low_delay1",
            "mode": "v2",
            "cost_model_cfg": {
                "mode": "v2",
                "round_trip_cost_pct": 0.1,
                "v2": {
                    "slippage_bps_base": 0.5,
                    "slippage_bps_vol_mult": 1.5,
                    "spread_bps": 0.5,
                    "delay_bars": 1,
                    "vol_proxy": "atr_pct",
                    "vol_lookback": 14,
                    "max_total_bps_per_side": 10.0,
                },
            },
            "slippage_pct": 0.0005,
        },
        {
            "label": "v2_high_delay1",
            "mode": "v2",
            "cost_model_cfg": {
                "mode": "v2",
                "round_trip_cost_pct": 0.1,
                "v2": {
                    "slippage_bps_base": 1.0,
                    "slippage_bps_vol_mult": 3.0,
                    "spread_bps": 1.0,
                    "delay_bars": 1,
                    "vol_proxy": "atr_pct",
                    "vol_lookback": 14,
                    "max_total_bps_per_side": 20.0,
                },
            },
            "slippage_pct": 0.0005,
        },
    ]

    for scenario in scenarios:
        result = run_backtest(
            frame=frame,
            strategy_name="Stage8CostSensitivity",
            symbol="BTC/USDT",
            max_hold_bars=1,
            stop_atr_multiple=20.0,
            take_profit_atr_multiple=20.0,
            round_trip_cost_pct=0.1,
            slippage_pct=float(scenario["slippage_pct"]),
            cost_model_cfg=scenario["cost_model_cfg"],
        )
        final_equity = float(result.equity_curve["equity"].iloc[-1]) if not result.equity_curve.empty else 0.0
        rows.append(
            {
                "scenario": str(scenario["label"]),
                "mode": str(scenario["mode"]),
                "trade_count": float(result.metrics["trade_count"]),
                "expectancy": float(result.metrics["expectancy"]),
                "profit_factor": float(result.metrics["profit_factor"]),
                "max_drawdown": float(result.metrics["max_drawdown"]),
                "final_equity": final_equity,
            }
        )

    output_df = pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)
    output_json = {
        "synthetic_data": True,
        "rows": output_df.to_dict(orient="records"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output_json, indent=2), encoding="utf-8")
    _write_markdown(args.output_md, output_df)
    print(f"wrote: {args.output_md}")
    print(f"wrote: {args.output_json}")


def _synthetic_frame(rows: int = 180) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = [100.0 + (idx * 0.05) for idx in range(rows)]
    high = [value + (0.6 if idx < rows // 2 else 2.0) for idx, value in enumerate(close)]
    low = [value - (0.6 if idx < rows // 2 else 2.0) for idx, value in enumerate(close)]
    signal = [0] * rows
    for idx in range(10, rows - 5, 8):
        signal[idx] = 1
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0,
            "atr_14": [1.0 if idx < rows // 2 else 3.0 for idx in range(rows)],
            "signal": signal,
        }
    )


def _write_markdown(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Stage-8 Cost Sensitivity")
    lines.append("")
    lines.append("Synthetic offline sweep comparing simple vs v2 cost model settings.")
    lines.append("")
    lines.append("| scenario | mode | trade_count | expectancy | profit_factor | max_drawdown | final_equity |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in df.to_dict(orient="records"):
        lines.append(
            f"| {row['scenario']} | {row['mode']} | {row['trade_count']:.0f} | {row['expectancy']:.6f} | "
            f"{row['profit_factor']:.6f} | {row['max_drawdown']:.6f} | {row['final_equity']:.2f} |"
        )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

