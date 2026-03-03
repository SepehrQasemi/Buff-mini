"""Stage-26.9.4 full data infrastructure audit orchestrator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import CANONICAL_DATA_DIR, DERIVED_DATA_DIR, RAW_DATA_DIR
from buffmini.data.derived_tf import get_timeframe
from buffmini.stage26.data_master_audit import build_master_summary, render_master_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Stage-26.9 data infrastructure audit")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--base", type=str, default="1m")
    parser.add_argument("--timeframes", type=str, default="1m,5m,15m,30m,1h,2h,4h,6h,12h,1d,1w,1M")
    parser.add_argument("--required-years", type=float, default=4.0)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--canonical-dir", type=Path, default=CANONICAL_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _run_script(command: list[str]) -> int:
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _parse_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _disk_usage_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += int(p.stat().st_size)
    return float(total) / (1024.0 * 1024.0)


def run_derived_sanity(
    *,
    symbols: list[str],
    exchange: str,
    canonical_dir: Path,
    derived_dir: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    targets = ["3h", "45m", "8h"]
    for symbol in symbols:
        for target in targets:
            first = get_timeframe(
                symbol=symbol,
                timeframe=target,
                exchange=exchange,
                canonical_dir=canonical_dir,
                derived_dir=derived_dir,
                drop_incomplete_last=True,
            )
            second = get_timeframe(
                symbol=symbol,
                timeframe=target,
                exchange=exchange,
                canonical_dir=canonical_dir,
                derived_dir=derived_dir,
                drop_incomplete_last=True,
            )
            checks.append(
                {
                    "symbol": symbol,
                    "target_tf": target,
                    "source_tf": first.source_timeframe,
                    "rows": int(first.frame.shape[0]),
                    "first_cache_hit": bool(first.cache_hit),
                    "second_cache_hit": bool(second.cache_hit),
                    "hash_match": bool(first.data_hash == second.data_hash),
                    "cache_path": str(first.cache_path),
                }
            )
    supported = sorted({row["target_tf"] for row in checks if bool(row.get("hash_match", False))})
    integrity = all(bool(row.get("hash_match", False)) for row in checks)
    return {"checks": checks, "supported": supported, "integrity_pass": bool(integrity)}


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    symbols = _parse_list(args.symbols)
    py = sys.executable

    raw_exit = _run_script(
        [
            py,
            "scripts/audit_raw_data.py",
            "--symbols",
            str(args.symbols),
            "--exchange",
            str(args.exchange),
            "--timeframe",
            str(args.base),
            "--required-years",
            str(args.required_years),
            "--data-dir",
            str(args.raw_dir),
            "--docs-dir",
            str(docs_dir),
        ]
    )

    _run_script(
        [
            py,
            "scripts/build_canonical_timeframes.py",
            "--symbols",
            str(args.symbols),
            "--exchange",
            str(args.exchange),
            "--base",
            str(args.base),
            "--timeframes",
            str(args.timeframes),
            "--raw-dir",
            str(args.raw_dir),
            "--canonical-dir",
            str(args.canonical_dir),
        ]
    )

    canonical_exit = _run_script(
        [
            py,
            "scripts/audit_canonical_data.py",
            "--symbols",
            str(args.symbols),
            "--exchange",
            str(args.exchange),
            "--base",
            str(args.base),
            "--timeframes",
            str(args.timeframes),
            "--raw-dir",
            str(args.raw_dir),
            "--canonical-dir",
            str(args.canonical_dir),
            "--docs-dir",
            str(docs_dir),
        ]
    )

    raw_payload = _read_json(docs_dir / "stage26_9_raw_data_audit.json")
    canonical_payload = _read_json(docs_dir / "stage26_9_canonical_audit.json")

    derived_payload = run_derived_sanity(
        symbols=symbols,
        exchange=str(args.exchange),
        canonical_dir=Path(args.canonical_dir),
        derived_dir=Path(args.derived_dir),
    )

    disk_usage = {
        "raw": _disk_usage_mb(Path(args.raw_dir)),
        "canonical": _disk_usage_mb(Path(args.canonical_dir)),
        "derived": _disk_usage_mb(Path(args.derived_dir)),
    }
    disk_usage["total"] = float(disk_usage["raw"] + disk_usage["canonical"] + disk_usage["derived"])

    summary = build_master_summary(
        raw_payload=raw_payload,
        canonical_payload=canonical_payload,
        derived_payload=derived_payload,
        disk_usage=disk_usage,
        raw_exit_code=raw_exit,
        canonical_exit_code=canonical_exit,
    )

    md_path = docs_dir / "stage26_9_data_master_report.md"
    json_path = docs_dir / "stage26_9_data_master_summary.json"
    md_path.write_text(render_master_md(summary), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"master_report: {md_path}")
    print(f"master_summary: {json_path}")


if __name__ == "__main__":
    main()
