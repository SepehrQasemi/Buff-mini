"""Run Stage-81 transfer and multi-asset reality summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.config import load_config
from buffmini.research.transfer import discover_transfer_symbols
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-81 transfer matrix summary")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _render(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage-81 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- execution_status: `{summary['execution_status']}`",
        f"- stage_role: `{summary['stage_role']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- primary_symbol: `{summary['primary_symbol']}`",
        f"- available_symbols: `{summary['available_symbols']}`",
        f"- additional_asset_available: `{summary['additional_asset_available']}`",
        f"- transfer_matrix_rows: `{summary['transfer_matrix_rows']}`",
        "",
        "## Transfer Classes",
    ]
    for key, value in sorted((summary.get("transfer_class_counts") or {}).items()):
        lines.append(f"- {key}: `{int(value)}`")
    lines.extend(["", "## Failure Diagnostics"])
    for key, value in sorted((summary.get("failure_diagnostics") or {}).items()):
        lines.append(f"- {key}: `{int(value)}`")
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage58 = _load_json(docs_dir / "stage58_summary.json")
    matrix = list(stage58.get("transfer_matrix", []))
    primary_symbol = str(stage58.get("primary_symbol", "")).strip()
    if not primary_symbol:
        primary_symbol = str((discover_transfer_symbols(cfg) or [""])[0])
    available_symbols = discover_transfer_symbols(cfg, primary_symbol=primary_symbol)
    class_counts = {str(key): int(value) for key, value in (stage58.get("transfer_class_counts") or {}).items()}
    diagnostics_hist: dict[str, int] = {}
    for row in matrix:
        for reason in list(row.get("diagnostics", [])):
            key = str(reason)
            diagnostics_hist[key] = diagnostics_hist.get(key, 0) + 1
    additional_asset_available = bool(len(available_symbols) >= 3)
    status = "SUCCESS" if matrix else "PARTIAL"
    summary = {
        "stage": "81",
        "status": status,
        "execution_status": "EXECUTED",
        "stage_role": "real_validation",
        "validation_state": "TRANSFER_MATRIX_READY" if matrix else "TRANSFER_MATRIX_INSUFFICIENT",
        "primary_symbol": primary_symbol,
        "available_symbols": available_symbols,
        "additional_asset_available": bool(additional_asset_available),
        "transfer_matrix_rows": int(len(matrix)),
        "transfer_class_counts": class_counts,
        "failure_diagnostics": diagnostics_hist,
        "artifact_path": str((docs_dir / "stage58_summary.json")).replace("\\", "/"),
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage81_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage81_report.md").write_text(_render(summary), encoding="utf-8")
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
