"""Stage-35.7 real-download runner with auth/plan gates and evidence reporting."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR

try:
    from scripts.run_stage35 import _check_min_coverage, _coinapi_defaults, _coverage_summary
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from run_stage35 import _check_min_coverage, _coinapi_defaults, _coverage_summary


REPORT_MD = Path("docs") / "stage35_7_report.md"
REPORT_JSON = Path("docs") / "stage35_7_report_summary.json"
AUTH_CLI_MD = Path("docs") / "stage35_7_auth_and_cli_fix.md"
USAGE_JSON = Path("docs") / "stage35_7_coinapi_usage.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-35.7 real CoinAPI download workflow")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--endpoints", type=str, default="funding,oi")
    parser.add_argument("--years", type=int, default=4)
    parser.add_argument("--increment-days", type=int, default=7)
    parser.add_argument("--max-requests", type=int, default=1500)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def _cmd(*parts: str) -> list[str]:
    return [sys.executable, *parts]


def _run(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": int(completed.returncode),
        "stdout": str(completed.stdout or ""),
        "stderr": str(completed.stderr or ""),
    }


def _snippet(text: str, *, limit: int = 1200) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]..."


def _parse_key_values(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in str(stdout or "").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _load_latest_usage_doc() -> dict[str, Any]:
    if not USAGE_JSON.exists():
        return {}
    payload = json.loads(USAGE_JSON.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _ensure_usage_doc(status: str) -> None:
    if USAGE_JSON.exists():
        return
    USAGE_JSON.parent.mkdir(parents=True, exist_ok=True)
    USAGE_JSON.write_text(
        json.dumps(
            {
                "runs": [],
                "latest": {
                    "status": str(status),
                    "total_requests_planned": 0,
                    "total_requests_selected": 0,
                    "total_requests_made": 0,
                    "status_code_counts": {},
                    "endpoints_hit": [],
                    "retry_counts": {"total": 0, "by_endpoint": {}},
                    "time_range": {"start": None, "end": None},
                    "rate_limit_sleep_ms_total": 0,
                    "estimated_credits_used": None,
                    "credits_estimation_mode": "UNKNOWN",
                },
                "totals": {"total_requests_planned": 0, "total_requests_made": 0, "run_count": 0},
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _write_auth_cli_doc() -> None:
    AUTH_CLI_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage-35.7 Auth and CLI Fix",
        "",
        "## Secret Resolution Order",
        "1. `COINAPI_KEY` from environment",
        "2. `secrets/coinapi_key.txt` (gitignored)",
        "3. `secrets/coinapi_key.json` with `COINAPI_KEY` key (gitignored)",
        "4. repo `.env` with `COINAPI_KEY=...` (gitignored)",
        "",
        "## Key Doctor Commands",
        "- status: `python scripts/coinapi_key_doctor.py --status`",
        "- wipe local secret files: `python scripts/coinapi_key_doctor.py --wipe-old`",
        "- write key interactively: `python scripts/coinapi_key_doctor.py --write`",
        "",
        "## CLI Aliases Added",
        "- `--download` => download action",
        "- `--plan` => planning action",
        "- `--verify` => auth probe (single lightweight request)",
        "- `--years N` => backfill range alias",
        "- `--budget-requests N` => alias for `--max-requests N`",
        "- endpoint aliases: `funding` => `funding_rates`, `oi` => `open_interest`",
        "",
        "## Missing Key Error",
        "When key is unavailable, downloader exits with:",
        "`COINAPI_KEY missing; use secrets/coinapi_key.txt (gitignored) or environment variable.`",
    ]
    AUTH_CLI_MD.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_report(payload: dict[str, Any]) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Stage-35.7 Report",
        "",
        f"- status: `{payload.get('status', '')}`",
        f"- key_source: `{payload.get('key_source', 'UNKNOWN')}`",
        f"- verify_ok: `{payload.get('verify_ok', False)}`",
        f"- plan_within_budget: `{payload.get('plan_within_budget', False)}`",
        f"- planned_requests: `{payload.get('planned_requests', 0)}`",
        f"- selected_requests: `{payload.get('selected_requests', 0)}`",
        f"- requests_made: `{payload.get('requests_made', 0)}`",
        f"- status_code_counts: `{payload.get('status_code_counts', {})}`",
        f"- coverage_ok: `{payload.get('coverage_ok', False)}`",
        "",
        "## Coverage Years",
    ]
    coverage_years = payload.get("coverage_years", {})
    if isinstance(coverage_years, dict) and coverage_years:
        for symbol, rows in sorted(coverage_years.items()):
            lines.append(f"- {symbol}: {rows}")
    else:
        lines.append("- not available")
    lines.extend(
        [
            "",
            "## Command Evidence",
            "### key_status stdout",
            "```text",
            _snippet(str(payload.get("key_status_stdout", ""))),
            "```",
            "### verify stdout",
            "```text",
            _snippet(str(payload.get("verify_stdout", ""))),
            "```",
            "### plan stdout",
            "```text",
            _snippet(str(payload.get("plan_stdout", ""))),
            "```",
            "### download stdout",
            "```text",
            _snippet(str(payload.get("download_stdout", ""))),
            "```",
            "### download stderr",
            "```text",
            _snippet(str(payload.get("download_stderr", ""))),
            "```",
        ]
    )
    REPORT_MD.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _coverage_payload(config_path: Path) -> tuple[dict[str, Any], bool, list[dict[str, Any]], float]:
    config = load_config(config_path)
    coinapi_cfg = _coinapi_defaults(config)
    coverage = _coverage_summary(config, coinapi_cfg)
    required = float(coinapi_cfg.get("require_min_coverage_years", 2.0))
    coverage_ok, missing = _check_min_coverage(coverage, required_years=required)
    return coverage, bool(coverage_ok), list(missing), required


def _extract_coverage_years(coverage: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for symbol, endpoints in (coverage.get("symbols", {}) or {}).items():
        if not isinstance(endpoints, dict):
            continue
        per_endpoint: dict[str, float] = {}
        for endpoint, row in endpoints.items():
            if not isinstance(row, dict):
                continue
            per_endpoint[str(endpoint)] = float(row.get("coverage_years", 0.0))
        out[str(symbol)] = per_endpoint
    return out


def run_stage35_real_download(args: argparse.Namespace) -> dict[str, Any]:
    _write_auth_cli_doc()

    status_cmd = _run(_cmd("scripts/coinapi_key_doctor.py", "--status"))
    status_stdout = status_cmd["stdout"].strip()
    key_source = "UNKNOWN"
    if "source=" in status_stdout:
        key_source = status_stdout.split("source=", 1)[1].strip()

    payload: dict[str, Any] = {
        "status": "INIT",
        "key_source": key_source,
        "verify_ok": False,
        "plan_within_budget": False,
        "planned_requests": 0,
        "selected_requests": 0,
        "requests_made": 0,
        "status_code_counts": {},
        "coverage_ok": False,
        "coverage_required_years": None,
        "coverage_years": {},
        "missing_coverage": [],
        "key_status_stdout": status_cmd["stdout"],
        "key_status_stderr": status_cmd["stderr"],
        "verify_stdout": "",
        "verify_stderr": "",
        "plan_stdout": "",
        "plan_stderr": "",
        "download_stdout": "",
        "download_stderr": "",
    }

    if status_cmd["returncode"] != 0 or status_stdout.startswith("MISSING"):
        payload["status"] = "AUTH_BLOCKED"
        _ensure_usage_doc(payload["status"])
        _write_report(payload)
        return payload

    common = [
        "--config",
        str(Path(args.config)),
        "--seed",
        str(int(args.seed)),
        "--symbols",
        str(args.symbols),
        "--endpoints",
        str(args.endpoints),
        "--years",
        str(int(args.years)),
        "--increment-days",
        str(int(args.increment_days)),
        "--max-requests",
        str(int(args.max_requests)),
        "--runs-dir",
        str(Path(args.runs_dir)),
    ]

    verify_cmd = _run(_cmd("scripts/update_coinapi_extras.py", "--verify", *common))
    payload["verify_stdout"] = verify_cmd["stdout"]
    payload["verify_stderr"] = verify_cmd["stderr"]
    payload["verify_ok"] = verify_cmd["returncode"] == 0
    if verify_cmd["returncode"] != 0:
        payload["status"] = "VERIFY_FAILED"
        usage_doc = _load_latest_usage_doc()
        latest = usage_doc.get("latest", {}) if isinstance(usage_doc, dict) else {}
        payload["status_code_counts"] = dict(latest.get("status_code_counts", {})) if isinstance(latest, dict) else {}
        _ensure_usage_doc(payload["status"])
        _write_report(payload)
        return payload

    plan_cmd = _run(_cmd("scripts/update_coinapi_extras.py", "--plan", *common))
    payload["plan_stdout"] = plan_cmd["stdout"]
    payload["plan_stderr"] = plan_cmd["stderr"]
    if plan_cmd["returncode"] != 0:
        payload["status"] = "PLAN_FAILED"
        _ensure_usage_doc(payload["status"])
        _write_report(payload)
        return payload
    plan_kv = _parse_key_values(plan_cmd["stdout"])
    planned = int(plan_kv.get("planned_count", "0") or 0)
    selected = int(plan_kv.get("selected_count", "0") or 0)
    truncated = str(plan_kv.get("truncated", "False")).lower() == "true"
    payload["planned_requests"] = planned
    payload["selected_requests"] = selected
    payload["plan_within_budget"] = not truncated
    if truncated:
        payload["status"] = "PLAN_OVER_BUDGET"
        _ensure_usage_doc(payload["status"])
        _write_report(payload)
        return payload

    download_cmd = _run(_cmd("scripts/update_coinapi_extras.py", "--download", *common))
    payload["download_stdout"] = download_cmd["stdout"]
    payload["download_stderr"] = download_cmd["stderr"]
    usage_doc = _load_latest_usage_doc()
    latest = usage_doc.get("latest", {}) if isinstance(usage_doc, dict) else {}
    if isinstance(latest, dict):
        payload["requests_made"] = int(latest.get("total_requests_made", 0))
        payload["status_code_counts"] = dict(latest.get("status_code_counts", {}))
    if download_cmd["returncode"] != 0:
        payload["status"] = "DOWNLOAD_FAILED"
        _ensure_usage_doc(payload["status"])
        _write_report(payload)
        return payload

    coverage, coverage_ok, missing, required_years = _coverage_payload(Path(args.config))
    payload["coverage_required_years"] = required_years
    payload["coverage_years"] = _extract_coverage_years(coverage)
    payload["coverage_ok"] = bool(coverage_ok)
    payload["missing_coverage"] = missing
    payload["status"] = "DOWNLOAD_COMPLETE" if coverage_ok else "INSUFFICIENT_COVERAGE"
    _ensure_usage_doc(payload["status"])
    _write_report(payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_stage35_real_download(args)
    print(f"status: {payload.get('status', '')}")
    print(f"report: {REPORT_MD.as_posix()}")
    print(f"summary: {REPORT_JSON.as_posix()}")


if __name__ == "__main__":
    main()
