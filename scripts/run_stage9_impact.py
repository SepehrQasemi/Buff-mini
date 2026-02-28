"""Run Stage-9 impact analysis from local OHLCV + derived funding/OI data."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.analysis.impact_analysis import analyze_symbol_impact, summarize_data_quality
from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR
from buffmini.data.derived_store import read_meta_json
from buffmini.data.features import calculate_features
from buffmini.data.store import build_data_store
from buffmini.validation.leakage_harness import run_registered_features_harness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-9 impact analysis")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
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
    timeframe = str(extras_cfg.get("timeframe", cfg["universe"]["timeframe"]))
    if timeframe != "1h":
        raise ValueError("Stage-9 impact analysis only supports 1h timeframe")

    store = build_data_store(
        backend=str(cfg.get("data", {}).get("backend", "parquet")),
        data_dir=args.data_dir,
    )

    impact_payload: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    quality_payload: dict[str, Any] = {}

    for symbol in symbols:
        frame = store.load_ohlcv(symbol=symbol, timeframe=timeframe)
        if frame.empty:
            raise RuntimeError(f"No OHLCV data for {symbol} {timeframe}")

        features = calculate_features(
            frame,
            config=cfg,
            symbol=symbol,
            timeframe=timeframe,
            derived_data_dir=args.derived_dir,
        )

        symbol_impact = analyze_symbol_impact(features=features, symbol=symbol, seed=int(args.seed), n_boot=int(args.n_boot))
        impact_payload[symbol] = symbol_impact
        all_rows.extend(symbol_impact["rows"])

        funding_meta = read_meta_json(
            kind="funding",
            symbol=symbol,
            timeframe=timeframe,
            data_dir=args.derived_dir,
        )
        oi_meta = read_meta_json(
            kind="open_interest",
            symbol=symbol,
            timeframe=timeframe,
            data_dir=args.derived_dir,
        )
        quality_payload[symbol] = summarize_data_quality(symbol=symbol, funding_meta=funding_meta, oi_meta=oi_meta)

    leakage = run_registered_features_harness(
        rows=420,
        seed=int(args.seed),
        shock_index=360,
        warmup_max=220,
        include_futures_extras=True,
    )

    rows_df = pd.DataFrame(all_rows)
    if rows_df.empty:
        strongest = []
    else:
        strongest = rows_df.reindex(rows_df["median_diff"].abs().sort_values(ascending=False).index).head(3)
        strongest = strongest.to_dict(orient="records")

    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    _write_impact_markdown(docs_dir / "stage9_impact_analysis.md", impact_payload, strongest)
    _write_data_quality_markdown(docs_dir / "stage9_data_quality.md", quality_payload)

    summary = {
        "data_quality": quality_payload,
        "leakage": {
            "features_checked": int(leakage["features_checked"]),
            "leaks_found": int(leakage["leaks_found"]),
        },
        "impact": {
            symbol: {
                "best_effect": payload["best_effect"],
                "ci": [
                    float(payload["best_effect"].get("ci_low", 0.0)),
                    float(payload["best_effect"].get("ci_high", 0.0)),
                ],
            }
            for symbol, payload in impact_payload.items()
        },
        "top_effects": strongest,
        "dsl_lite": {
            "trade_count_ratio_bounds_ok": False,
        },
        "runtime_seconds": round(time.time() - started, 3),
    }

    (docs_dir / "stage9_report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Stage-9 Report",
        "",
        "## What Was Added",
        "- Funding and open-interest derived ingestion/store",
        "- Leakage-safe futures extras features",
        "- Stage-9 impact analysis tables and quality report",
        "",
        "## Leak-Proof Evidence",
        f"- features_checked: `{summary['leakage']['features_checked']}`",
        f"- leaks_found: `{summary['leakage']['leaks_found']}`",
        "",
        "## Impact Evidence",
    ]
    if strongest:
        for row in strongest:
            report_lines.append(
                "- "
                f"{row['symbol']} | {row['condition']} | {row['horizon']} | "
                f"median_diff={float(row['median_diff']):.6f} | "
                f"CI=[{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}]"
            )
    else:
        report_lines.append("- no meaningful bias detected")

    report_lines.extend(
        [
            "",
            "## Runtime Notes",
            f"- impact_analysis_runtime_seconds: `{summary['runtime_seconds']}`",
            "- download runtime tracked via scripts/update_futures_extras.py execution",
        ]
    )

    (docs_dir / "stage9_report.md").write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    print("wrote: docs/stage9_impact_analysis.md")
    print("wrote: docs/stage9_data_quality.md")
    print("wrote: docs/stage9_report_summary.json")
    print("wrote: docs/stage9_report.md")


def _write_impact_markdown(path: Path, payload: dict[str, Any], strongest: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Stage-9 Impact Analysis")
    lines.append("")
    lines.append("Conditional forward-return analysis from funding/OI regimes.")
    lines.append("")

    for symbol, item in payload.items():
        lines.append(f"## {symbol}")
        lines.append("")
        lines.append(
            f"- corr(funding_z_30, forward_return_24h): `{float(item['corr_funding_z30_vs_fwd24']):.6f}`"
        )
        lines.append(f"- corr(oi_z_30, forward_return_24h): `{float(item['corr_oi_z30_vs_fwd24']):.6f}`")
        lines.append("")
        lines.append(
            "| condition | horizon | count_cond | count_base | median_cond | median_base | median_diff | ci_low | ci_high |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in item["rows"]:
            lines.append(
                f"| {row['condition']} | {row['horizon']} | {int(row['count_condition'])} | {int(row['count_base'])} | "
                f"{float(row['median_condition']):.6f} | {float(row['median_base']):.6f} | {float(row['median_diff']):.6f} | "
                f"{float(row['ci_low']):.6f} | {float(row['ci_high']):.6f} |"
            )
        lines.append("")

    lines.append("## Strongest Effects")
    if strongest:
        for row in strongest:
            lines.append(
                "- "
                f"{row['symbol']} | {row['condition']} | {row['horizon']} | "
                f"median_diff={float(row['median_diff']):.6f} | "
                f"CI=[{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}]"
            )
    else:
        lines.append("- no meaningful bias detected")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_data_quality_markdown(path: Path, quality: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Stage-9 Data Quality")
    lines.append("")
    lines.append("| symbol | funding_rows | funding_start | funding_end | funding_gaps | oi_rows | oi_start | oi_end | oi_gaps |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: | --- | --- | ---: |")

    for symbol, row in quality.items():
        funding = row["funding"]
        oi = row["open_interest"]
        lines.append(
            f"| {symbol} | {int(funding['row_count'])} | {funding['start_ts']} | {funding['end_ts']} | {int(funding['gaps_count'])} | "
            f"{int(oi['row_count'])} | {oi['start_ts']} | {oi['end_ts']} | {int(oi['gaps_count'])} |"
        )

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
