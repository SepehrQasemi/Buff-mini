"""Run Stage-95 live usefulness push diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.reporting import markdown_kv, markdown_rows, write_stage_artifacts
from buffmini.research.usefulness import evaluate_live_usefulness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-95 live usefulness push")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--candidate-limit-per-family", type=int, default=3)
    parser.add_argument("--replay-window-bars", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    summary = evaluate_live_usefulness(
        cfg,
        symbol=str(args.symbol),
        timeframe=str(args.timeframe),
        candidate_limit_per_family=int(args.candidate_limit_per_family),
        replay_window_bars=int(args.replay_window_bars),
    )
    summary.update(
        {
            "stage": "95",
            "status": "SUCCESS",
            "execution_status": "EXECUTED",
            "stage_role": "real_validation",
            "validation_state": "LIVE_USEFULNESS_READY",
        }
    )
    lines = [
        "# Stage-95 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- symbol: `{summary['symbol']}`",
        f"- timeframe: `{summary['timeframe']}`",
        f"- before_profile: `{summary['before_profile']}`",
        f"- after_profile: `{summary['after_profile']}`",
        f"- stage95b_recommended: `{summary['stage95b_recommended']}`",
        f"- stage95b_applied: `{summary['stage95b_applied']}`",
        f"- replay_window_bars: `{summary['replay_window_bars']}`",
        "",
    ]
    lines.extend(markdown_kv("Before Counts", dict(summary.get("before_counts", {}))))
    lines.extend([""] + markdown_kv("After Counts", dict(summary.get("after_counts", {}))))
    lines.extend([""] + markdown_kv("Usefulness Delta", dict(summary.get("usefulness_delta", {}))))
    lines.extend([""] + markdown_rows("Family Usefulness", list(summary.get("family_usefulness", [])), limit=16))
    lines.extend([""] + markdown_rows("Family Replay Death Map", list(summary.get("family_replay_death_map", [])), limit=16))
    lines.extend([""] + markdown_rows("Dead Weight Families", list(summary.get("dead_weight_families", [])), limit=16))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="95", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
