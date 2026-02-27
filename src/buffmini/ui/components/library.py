"""Stage-5 Strategy Library helpers."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from buffmini.constants import PROJECT_ROOT, RUNS_DIR
from buffmini.ui.components.artifacts import load_stage3_3_artifacts
from buffmini.utils.hashing import stable_hash


LIBRARY_DIR = PROJECT_ROOT / "library"
LIBRARY_INDEX_PATH = LIBRARY_DIR / "index.json"


def ensure_library_layout(library_dir: Path = LIBRARY_DIR) -> None:
    """Ensure library root/index/strategies folder exists."""

    base = Path(library_dir)
    (base / "strategies").mkdir(parents=True, exist_ok=True)
    if not (base / "index.json").exists():
        (base / "index.json").write_text(json.dumps({"strategies": []}, indent=2), encoding="utf-8")


def load_library_index(library_dir: Path = LIBRARY_DIR) -> dict[str, Any]:
    """Load library index JSON safely."""

    ensure_library_layout(library_dir)
    path = Path(library_dir) / "index.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {"strategies": []}
    if "strategies" not in payload or not isinstance(payload["strategies"], list):
        payload = {"strategies": []}
    return payload


def save_library_index(index_payload: dict[str, Any], library_dir: Path = LIBRARY_DIR) -> None:
    """Atomically persist library index JSON."""

    ensure_library_layout(library_dir)
    path = Path(library_dir) / "index.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def generate_strategy_id(
    method: str,
    leverage: float,
    symbols: list[str],
    timeframe: str,
    date_label: str,
) -> str:
    """Generate deterministic strategy_id from metadata signature."""

    base = {
        "method": str(method),
        "leverage": float(leverage),
        "symbols": sorted([str(item) for item in symbols]),
        "timeframe": str(timeframe),
        "date_label": str(date_label),
    }
    return f"{str(method).lower()}_{str(timeframe)}_{stable_hash(base, length=10)}"


def export_run_to_library(
    run_id: str,
    display_name: str | None = None,
    runs_dir: Path = RUNS_DIR,
    library_dir: Path = LIBRARY_DIR,
) -> dict[str, Any]:
    """Export compact strategy metadata/spec from run artifacts into library."""

    ensure_library_layout(library_dir)
    run_dir = Path(runs_dir) / str(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_id}")

    pipeline_summary = _safe_json(run_dir / "pipeline_summary.json")
    stage3_run_id = str(pipeline_summary.get("stage3_3_run_id", "") or "")
    stage2_run_id = str(pipeline_summary.get("stage2_run_id", "") or "")
    stage1_run_id = str(pipeline_summary.get("stage1_run_id", "") or "")
    chosen_method = str(pipeline_summary.get("chosen_method", "equal"))
    chosen_leverage = float(pipeline_summary.get("chosen_leverage", 1.0))

    selector_summary = {}
    if stage3_run_id:
        selector_summary, _ = load_stage3_3_artifacts(Path(runs_dir) / stage3_run_id)
        selector_summary = selector_summary.get("summary", {})
        if selector_summary:
            overall = selector_summary.get("overall_choice", {})
            chosen_method = str(overall.get("method", chosen_method))
            chosen_leverage = float(overall.get("chosen_leverage", chosen_leverage))
            stage2_run_id = str(selector_summary.get("stage2_run_id", stage2_run_id))
            stage1_run_id = str(selector_summary.get("stage1_run_id", stage1_run_id))

    stage2_summary = _safe_json(Path(runs_dir) / stage2_run_id / "portfolio_summary.json") if stage2_run_id else {}
    symbols = list((stage2_summary.get("universe") or {}).get("symbols", []))
    if not symbols:
        symbols = ["BTC/USDT", "ETH/USDT"]
    timeframe = str((stage2_summary.get("universe") or {}).get("timeframe", "1h"))
    date_label = datetime.now(timezone.utc).strftime("%Y%m%d")
    strategy_id = generate_strategy_id(
        method=chosen_method,
        leverage=chosen_leverage,
        symbols=symbols,
        timeframe=timeframe,
        date_label=date_label,
    )
    strategy_dir = Path(library_dir) / "strategies" / strategy_id
    strategy_dir.mkdir(parents=True, exist_ok=True)

    docs_candidates = [
        PROJECT_ROOT / "docs" / "trading_spec.md",
        run_dir / "trading_spec.md",
    ]
    checklist_candidates = [
        PROJECT_ROOT / "docs" / "paper_trading_checklist.md",
        run_dir / "paper_trading_checklist.md",
    ]
    spec_src = _first_existing(docs_candidates)
    checklist_src = _first_existing(checklist_candidates)
    if spec_src:
        shutil.copyfile(spec_src, strategy_dir / "strategy_spec.md")
    else:
        (strategy_dir / "strategy_spec.md").write_text("# Strategy Spec\n\nMissing source spec.\n", encoding="utf-8")
    if checklist_src:
        shutil.copyfile(checklist_src, strategy_dir / "paper_trading_checklist.md")
    else:
        (strategy_dir / "paper_trading_checklist.md").write_text("# Paper Checklist\n\nMissing source checklist.\n", encoding="utf-8")

    weights_sources = []
    if stage2_run_id:
        weights_sources.extend(
            [
                Path(runs_dir) / stage2_run_id / "weights_equal.csv",
                Path(runs_dir) / stage2_run_id / "weights_vol.csv",
                Path(runs_dir) / stage2_run_id / "weights_corr_min.csv",
            ]
        )
    weight_src = _first_existing(weights_sources)
    if weight_src:
        shutil.copyfile(weight_src, strategy_dir / "weights.csv")

    params = {
        "method": chosen_method,
        "leverage": float(chosen_leverage),
        "symbols": symbols,
        "timeframe": timeframe,
        "execution_mode": str(_safe_json(PROJECT_ROOT / "configs" / "default.yaml").get("execution", {}).get("mode", "net")),
    }
    (strategy_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    origin = {
        "run_id": str(run_id),
        "stage1_run_id": stage1_run_id,
        "stage2_run_id": stage2_run_id,
        "stage3_3_run_id": stage3_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if selector_summary:
        origin["config_hash"] = selector_summary.get("config_hash")
        origin["data_hash"] = selector_summary.get("data_hash")
    (strategy_dir / "origin.json").write_text(json.dumps(origin, indent=2), encoding="utf-8")

    display = display_name.strip() if display_name and display_name.strip() else f"{chosen_method.upper()} {chosen_leverage}x"
    strategy_card = {
        "strategy_id": strategy_id,
        "display_name": display,
        "method": chosen_method,
        "leverage": float(chosen_leverage),
        "symbols": symbols,
        "timeframe": timeframe,
        "execution_mode": params["execution_mode"],
        "origin_run_id": str(run_id),
    }
    (strategy_dir / "strategy_card.json").write_text(json.dumps(strategy_card, indent=2), encoding="utf-8")

    index_payload = load_library_index(library_dir)
    strategies = [item for item in index_payload.get("strategies", []) if item.get("strategy_id") != strategy_id]
    strategies.append(strategy_card)
    index_payload["strategies"] = sorted(strategies, key=lambda item: item.get("display_name", ""))
    save_library_index(index_payload, library_dir)
    return strategy_card


def load_strategy_params(strategy_id: str, library_dir: Path = LIBRARY_DIR) -> dict[str, Any]:
    """Load params.json for selected strategy card."""

    params_path = Path(library_dir) / "strategies" / str(strategy_id) / "params.json"
    if not params_path.exists():
        return {}
    return _safe_json(params_path)


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        # Minimal safe parser for YAML-like default config readback.
        import yaml

        parsed = yaml.safe_load(text) or {}
        return parsed if isinstance(parsed, dict) else {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None

