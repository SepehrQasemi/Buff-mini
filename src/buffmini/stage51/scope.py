"""Stage-51 research scope freeze and budget framework."""

from __future__ import annotations

from typing import Any

from buffmini.utils.hashing import stable_hash


DEFAULT_SETUP_FAMILIES: tuple[str, ...] = (
    "structure_pullback_continuation",
    "liquidity_sweep_reversal",
    "squeeze_flow_breakout",
)
DEFAULT_DISCOVERY_TIMEFRAMES: tuple[str, ...] = ("15m", "30m", "1h", "2h", "4h")
ALLOWED_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1M")


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return out


def resolve_research_scope(config: dict[str, Any]) -> dict[str, Any]:
    scope_cfg = dict(config.get("research_scope", {}))
    primary_symbols = _dedupe([str(v) for v in scope_cfg.get("primary_symbols", ["BTC/USDT"])]) or ["BTC/USDT"]
    discovery_timeframes = _dedupe([str(v).lower() for v in scope_cfg.get("discovery_timeframes", list(DEFAULT_DISCOVERY_TIMEFRAMES))])
    if not discovery_timeframes:
        discovery_timeframes = list(DEFAULT_DISCOVERY_TIMEFRAMES)
    invalid_tfs = sorted(tf for tf in discovery_timeframes if tf not in set(ALLOWED_TIMEFRAMES))
    if invalid_tfs:
        raise ValueError(f"Unsupported discovery timeframes: {invalid_tfs}")

    promotion_timeframes = int(scope_cfg.get("promotion_timeframes", 2))
    final_validation_timeframes = int(scope_cfg.get("final_validation_timeframes", 1))
    promotion_timeframes = max(1, min(promotion_timeframes, len(discovery_timeframes)))
    final_validation_timeframes = max(1, min(final_validation_timeframes, promotion_timeframes))

    families = _dedupe([str(v) for v in scope_cfg.get("active_setup_families", list(DEFAULT_SETUP_FAMILIES))]) or list(DEFAULT_SETUP_FAMILIES)
    expansion_rules = dict(scope_cfg.get("expansion_rules", {}))
    transfer_symbol = str(expansion_rules.get("transfer_symbol", "ETH/USDT")).strip() or "ETH/USDT"
    require_stage57_pass = bool(expansion_rules.get("require_stage57_pass", True))

    oi_cfg = (
        config.get("data", {})
        .get("futures_extras", {})
        .get("open_interest", {})
    )
    oi_core_enabled = bool(expansion_rules.get("oi_core_enabled", False))
    oi_short_horizon_only = bool(oi_cfg.get("short_horizon_only", False))
    oi_short_horizon_max = str(oi_cfg.get("short_horizon_max", "30m"))

    resolved = {
        "primary_symbols": primary_symbols,
        "discovery_timeframes": discovery_timeframes,
        "promotion_timeframes": promotion_timeframes,
        "final_validation_timeframes": final_validation_timeframes,
        "active_setup_families": families,
        "expansion_rules": {
            "transfer_symbol": transfer_symbol,
            "require_stage57_pass": require_stage57_pass,
            "oi_core_enabled": oi_core_enabled,
            "oi_short_horizon_only": oi_short_horizon_only,
            "oi_short_horizon_max": oi_short_horizon_max,
        },
    }
    return resolved


def resolve_budget_mode(config: dict[str, Any]) -> dict[str, Any]:
    budget_cfg = dict(config.get("budget_mode", {}))
    selected = str(budget_cfg.get("selected", "search")).strip().lower() or "search"
    if selected not in {"smoke", "search", "validate", "full_audit"}:
        raise ValueError(f"Unsupported budget mode: {selected}")
    modes = {}
    for name in ("smoke", "search", "validate", "full_audit"):
        item = dict(budget_cfg.get(name, {}))
        modes[name] = {
            "candidate_limit": int(item.get("candidate_limit", 0)),
            "stage_a_limit": int(item.get("stage_a_limit", 0)),
            "stage_b_limit": int(item.get("stage_b_limit", 0)),
            "micro_replay_limit": int(item.get("micro_replay_limit", 0)),
            "full_replay_limit": int(item.get("full_replay_limit", 0)),
            "walkforward_limit": int(item.get("walkforward_limit", 0)),
            "monte_carlo_limit": int(item.get("monte_carlo_limit", 0)),
            "max_runtime_seconds": int(item.get("max_runtime_seconds", 0)),
        }
    active = dict(modes[selected])
    return {"selected": selected, "active": active, "modes": modes}


def build_stage51_summary(config: dict[str, Any]) -> dict[str, Any]:
    scope = resolve_research_scope(config)
    budget = resolve_budget_mode(config)
    payload = {
        "stage": "51",
        "status": "SUCCESS",
        "primary_symbols": list(scope["primary_symbols"]),
        "discovery_timeframes": list(scope["discovery_timeframes"]),
        "promotion_timeframes": int(scope["promotion_timeframes"]),
        "final_validation_timeframes": int(scope["final_validation_timeframes"]),
        "active_setup_families": list(scope["active_setup_families"]),
        "expansion_rules": dict(scope["expansion_rules"]),
        "budget_mode_selected": str(budget["selected"]),
        "budget_active": dict(budget["active"]),
        "budget_modes": dict(budget["modes"]),
    }
    payload["summary_hash"] = stable_hash(payload, length=16)
    return payload


def render_stage51_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage-51 Research Scope Freeze Report",
        "",
        f"- status: `{summary.get('status', '')}`",
        f"- primary_symbols: `{summary.get('primary_symbols', [])}`",
        f"- discovery_timeframes: `{summary.get('discovery_timeframes', [])}`",
        f"- promotion_timeframes: `{summary.get('promotion_timeframes', 0)}`",
        f"- final_validation_timeframes: `{summary.get('final_validation_timeframes', 0)}`",
        f"- active_setup_families: `{summary.get('active_setup_families', [])}`",
        f"- budget_mode_selected: `{summary.get('budget_mode_selected', '')}`",
        "",
        "## Expansion Rules",
    ]
    for key, value in sorted(dict(summary.get("expansion_rules", {})).items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Active Budget"])
    for key, value in sorted(dict(summary.get("budget_active", {})).items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", f"- summary_hash: `{summary.get('summary_hash', '')}`"])
    return "\n".join(lines).strip() + "\n"
