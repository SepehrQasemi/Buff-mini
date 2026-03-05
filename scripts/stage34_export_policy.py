"""Export latest Stage-34 policy artifact into local strategy library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage-34 policy to library")
    parser.add_argument("--policy-path", type=str, default="")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--library-dir", type=Path, default=Path("library/strategies"))
    return parser.parse_args()


def _latest_policy_path(runs_dir: Path) -> Path:
    items = sorted(Path(runs_dir).glob("*_stage34_policy/stage34/policy/policy_snapshot.json"))
    if not items:
        raise FileNotFoundError("No stage34 policy snapshot found")
    return items[-1]


def main() -> None:
    args = parse_args()
    policy_path = Path(str(args.policy_path).strip()) if str(args.policy_path).strip() else _latest_policy_path(Path(args.runs_dir))
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    strategy_id = f"stage34_{stable_hash({'policy_id': payload.get('policy_id', ''), 'model': payload.get('model_name', '')}, length=10)}"
    out_dir = Path(args.library_dir) / strategy_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "policy.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"strategy_id: {strategy_id}")
    print(f"policy_path: {out_path}")


if __name__ == "__main__":
    main()
