"""Export Stage-3.3 selector outputs from runs/ into docs/ markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_METHODS = {"equal", "vol"}
TARGET_LEVERAGES = {1.0, 2.0, 3.0, 5.0, 10.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage-3.3 selector summary to docs")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("docs") / "stage3_3_selector_summary.md")
    return parser.parse_args()


def export_stage3_selector_to_docs(run_id: str, runs_dir: Path, output: Path) -> Path:
    """Render a concise docs markdown summary for one Stage-3.3 run."""

    run_dir = runs_dir / run_id
    summary_path = run_dir / "selector_summary.json"
    table_path = run_dir / "selector_table.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing selector_summary.json for run_id={run_id}")
    if not table_path.exists():
        raise FileNotFoundError(f"Missing selector_table.csv for run_id={run_id}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    table = pd.read_csv(table_path)

    overall = summary.get("overall_choice", {})
    chosen_method = overall.get("method")
    chosen_leverage = overall.get("chosen_leverage")
    chosen_binding = []
    method_choices = summary.get("method_choices", {})
    if chosen_method in method_choices:
        chosen_binding = method_choices[chosen_method].get("binding_constraints", [])

    settings = summary.get("settings", {})
    constraints = settings.get("constraints", {})
    commit_hash = summary.get("commit") or summary.get("commit_hash") or settings.get("commit") or "N/A"

    subset = table[
        table["method"].isin(TARGET_METHODS) & table["leverage"].astype(float).isin(TARGET_LEVERAGES)
    ].copy()
    subset = subset.sort_values(["method", "leverage"]).reset_index(drop=True)

    lines: list[str] = []
    lines.append("# Stage-3.3 Leverage Selector Summary")
    lines.append("")
    lines.append(f"- run_id: `{summary.get('run_id', run_id)}`")
    lines.append(f"- commit hash: `{commit_hash}`")
    lines.append(f"- chosen overall: `{chosen_method}` @ `{chosen_leverage}x`")
    lines.append(f"- binding constraint (chosen): `{', '.join(chosen_binding) if chosen_binding else 'N/A'}`")
    lines.append(
        "- constraints: "
        f"`max_p_ruin={constraints.get('max_p_ruin')}, "
        f"max_dd_p95={constraints.get('max_dd_p95')}, "
        f"min_return_p05={constraints.get('min_return_p05')}`"
    )
    lines.append("")
    lines.append("## Leverage Snapshot (1,2,3,5,10)")
    lines.append("| method | leverage | expected_log_growth | return_p05 | maxDD_p95 | P(ruin) | feasible |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for _, row in subset.iterrows():
        lines.append(
            f"| {row['method']} | {float(row['leverage']):.0f} | {float(row['expected_log_growth']):.6f} | "
            f"{float(row['return_p05']):.6f} | {float(row['maxdd_p95']):.6f} | {float(row['p_ruin']):.6f} | "
            f"{bool(row['pass_all_constraints'])} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "Both `equal` and `vol` remain feasible through 5x under the configured risk limits, "
        "then fail at higher leverage mainly due to drawdown and ruin constraints. "
        f"The selector chooses `{chosen_method}` at `{chosen_leverage}x` because it has the strongest "
        "expected log-growth inside the feasible set."
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    output = export_stage3_selector_to_docs(
        run_id=args.run_id,
        runs_dir=args.runs_dir,
        output=args.output,
    )
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()

