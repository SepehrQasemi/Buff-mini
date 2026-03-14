from pathlib import Path

from buffmini.diagnostics.full_trace import write_full_trace_report


def test_full_trace_report_writes(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    runs_dir = tmp_path / "runs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "stage52_summary.json").write_text(
        '{"stage":"52","status":"SUCCESS","stage28_run_id":"dummy"}',
        encoding="utf-8",
    )
    (docs_dir / "stage57_summary.json").write_text(
        '{"stage":"57","status":"PARTIAL","verdict":"PARTIAL"}',
        encoding="utf-8",
    )
    payload = write_full_trace_report(
        docs_dir=docs_dir,
        runs_dir=runs_dir,
        config_path=Path("configs/default.yaml"),
    )
    assert (docs_dir / "full_trace_summary.json").exists()
    assert (docs_dir / "full_trace_report.md").exists()
    assert payload.get("summary_hash")

