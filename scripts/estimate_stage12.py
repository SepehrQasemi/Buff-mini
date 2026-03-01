"""Stage-12 runtime preflight estimator CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.perf.stage12_estimator import estimate_stage12_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Stage-12 runtime from bench metrics")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--bench-json", type=Path, default=None, help="Path to runs/*/perf_meta.json")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bench_path = args.bench_json or _latest_perf_meta(args.runs_dir)
    if bench_path is None or not Path(bench_path).exists():
        raise FileNotFoundError("No perf_meta.json found. Run scripts/bench_engine_stage11_5.py first.")
    bench_metrics = json.loads(Path(bench_path).read_text(encoding="utf-8"))
    summary = estimate_stage12_runtime(config=config, bench_metrics=bench_metrics)

    print(f"bench_json: {bench_path}")
    print(f"estimated_total_seconds: {summary['estimated_total_seconds']:.2f}")
    print(f"estimated_total_minutes: {summary['estimated_total_minutes']:.2f}")
    print(f"recommendation: {summary['recommendation']}")
    for tf, seconds in summary["per_timeframe_seconds"].items():
        print(f"  {tf}: {seconds:.2f}s")
    if float(summary["estimated_total_minutes"]) > 90.0:
        print("warning: estimated runtime > 90 minutes")


def _latest_perf_meta(runs_dir: Path) -> Path | None:
    candidates = sorted(Path(runs_dir).glob("*_stage11_5_bench/perf_meta.json"))
    if not candidates:
        return None
    return candidates[-1]


if __name__ == "__main__":
    main()
