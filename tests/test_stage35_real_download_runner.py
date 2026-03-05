from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts import run_stage35_real_download


def _args(tmp_path: Path) -> Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    return Namespace(
        config=repo_root / "configs" / "default.yaml",
        seed=42,
        symbols="BTC/USDT,ETH/USDT",
        endpoints="funding,oi",
        years=4,
        increment_days=7,
        max_requests=1500,
        runs_dir=tmp_path / "runs",
    )


def test_runner_writes_auth_blocked_report_when_key_missing(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)

    calls = {"count": 0}

    def fake_run(cmd: list[str]) -> dict[str, object]:
        calls["count"] += 1
        return {"cmd": cmd, "returncode": 1, "stdout": "MISSING\n", "stderr": ""}

    monkeypatch.setattr(run_stage35_real_download, "_run", fake_run)
    payload = run_stage35_real_download.run_stage35_real_download(_args(tmp_path))
    assert payload["status"] == "AUTH_BLOCKED"
    assert calls["count"] == 1
    assert (tmp_path / "docs" / "stage35_7_report.md").exists()
    assert (tmp_path / "docs" / "stage35_7_report_summary.json").exists()
    assert (tmp_path / "docs" / "stage35_7_auth_and_cli_fix.md").exists()


def test_runner_stops_before_download_on_plan_over_budget(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)

    outputs = [
        {"returncode": 0, "stdout": "OK source=SECRETS_TXT\n", "stderr": ""},
        {"returncode": 0, "stdout": "verify_status: OK\n", "stderr": ""},
        {
            "returncode": 0,
            "stdout": "planned_count: 100\nselected_count: 20\ntruncated: True\nplan_path: runs/x/coinapi/plan.json\n",
            "stderr": "",
        },
    ]
    idx = {"value": 0}

    def fake_run(cmd: list[str]) -> dict[str, object]:
        out = outputs[idx["value"]]
        idx["value"] += 1
        return {"cmd": cmd, **out}

    monkeypatch.setattr(run_stage35_real_download, "_run", fake_run)
    payload = run_stage35_real_download.run_stage35_real_download(_args(tmp_path))
    assert payload["status"] == "PLAN_OVER_BUDGET"
    assert payload["plan_within_budget"] is False
    assert payload["planned_requests"] == 100
    assert payload["selected_requests"] == 20

