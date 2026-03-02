"""Stage-24 deterministic sizing math."""

from __future__ import annotations

from typing import Any

from buffmini.stage23.rejects import EXECUTION_REJECT_REASONS


def compute_risk_pct(
    *,
    equity: float,
    dd: float,
    losing_streak: int,
    cfg: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    stage24 = _resolve_stage24_cfg(cfg)
    sizing = dict(stage24.get("sizing", {}))
    ladder = dict(sizing.get("risk_ladder", {}))
    clamps = dict(sizing.get("clamps", {}))

    r_min = float(ladder.get("r_min", 0.02))
    r_max = float(ladder.get("r_max", 0.20))
    risk_user = sizing.get("risk_pct_user", None)
    equity_safe = max(float(equity), 1e-12)

    if risk_user is not None:
        base = _clamp(float(risk_user), r_min, r_max)
    elif bool(ladder.get("enabled", True)):
        r_ref = float(ladder.get("r_ref", 0.08))
        e_ref = max(float(ladder.get("e_ref", 1000.0)), 1e-12)
        k = max(float(ladder.get("k", 0.5)), 0.0)
        raw = float(r_ref * ((e_ref / equity_safe) ** k))
        base = _clamp(raw, r_min, r_max)
    else:
        base = _clamp(float(ladder.get("r_ref", 0.08)), r_min, r_max)

    dd_value = max(0.0, float(dd))
    dd_soft = float(clamps.get("dd_soft", 0.10))
    dd_hard = float(clamps.get("dd_hard", 0.20))
    dd_mult = 1.0
    if dd_value >= dd_hard:
        dd_mult = float(clamps.get("dd_hard_mult", 0.4))
    elif dd_value >= dd_soft:
        dd_mult = float(clamps.get("dd_soft_mult", 0.7))

    streak = max(0, int(losing_streak))
    streak_soft = int(clamps.get("losing_streak_soft", 3))
    streak_hard = int(clamps.get("losing_streak_hard", 5))
    streak_mult = 1.0
    if streak >= streak_hard:
        streak_mult = float(clamps.get("streak_hard_mult", 0.4))
    elif streak >= streak_soft:
        streak_mult = float(clamps.get("streak_soft_mult", 0.7))

    used = _clamp(float(base * dd_mult * streak_mult), r_min, r_max)
    components = {
        "base": float(base),
        "dd_mult": float(dd_mult),
        "streak_mult": float(streak_mult),
        "used": float(used),
        "r_min": float(r_min),
        "r_max": float(r_max),
    }
    return float(used), components


def compute_notional_risk_pct(
    *,
    equity: float,
    risk_pct_used: float,
    stop_distance_pct: float,
    cost_rt_pct: float,
    constraints_cfg: dict[str, Any],
) -> tuple[float, str, str, dict[str, Any]]:
    denom = float(stop_distance_pct) + float(cost_rt_pct)
    if denom <= 0:
        return 0.0, "INVALID", "STOP_INVALID", {"denom": float(denom), "stop_distance_pct": float(stop_distance_pct), "cost_rt_pct": float(cost_rt_pct)}
    notional_raw = float(max(0.0, equity) * max(0.0, float(risk_pct_used)) / denom)
    return _apply_notional_constraints(notional_raw=notional_raw, equity=float(equity), constraints_cfg=constraints_cfg, input_type="risk_pct")


def compute_notional_alloc_pct(
    *,
    equity: float,
    alloc_pct: float,
    constraints_cfg: dict[str, Any],
) -> tuple[float, str, str, dict[str, Any]]:
    notional_raw = float(max(0.0, equity) * max(0.0, float(alloc_pct)))
    return _apply_notional_constraints(notional_raw=notional_raw, equity=float(equity), constraints_cfg=constraints_cfg, input_type="alloc_pct")


def cost_rt_pct_from_config(cfg: dict[str, Any]) -> float:
    """Round-trip cost as fraction (e.g., 0.001 means 0.1%)."""

    cost_model = dict(cfg.get("cost_model", {}) if isinstance(cfg, dict) else {})
    costs = dict(cfg.get("costs", {}) if isinstance(cfg, dict) else {})
    if cost_model:
        return float(float(cost_model.get("round_trip_cost_pct", costs.get("round_trip_cost_pct", 0.1))) / 100.0)
    return float(float(costs.get("round_trip_cost_pct", 0.1)) / 100.0)


def _apply_notional_constraints(
    *,
    notional_raw: float,
    equity: float,
    constraints_cfg: dict[str, Any],
    input_type: str,
) -> tuple[float, str, str, dict[str, Any]]:
    min_notional = float(constraints_cfg.get("min_trade_notional", 10.0))
    allow_bump = bool(constraints_cfg.get("allow_size_bump_to_min_notional", True))
    max_notional_pct = float(constraints_cfg.get("max_notional_pct_of_equity", 1.0))
    max_notional = float(max(0.0, max_notional_pct * max(0.0, equity)))

    if notional_raw <= 0:
        return 0.0, "INVALID", "SIZE_ZERO", _details(
            input_type=input_type,
            notional_raw=notional_raw,
            notional_capped=0.0,
            max_notional=max_notional,
            min_notional=min_notional,
            bumped=False,
        )

    notional_capped = float(min(notional_raw, max_notional)) if max_notional > 0 else 0.0
    bumped = False
    reason = ""

    if notional_capped < min_notional:
        if allow_bump and min_notional <= max_notional:
            notional_capped = float(min_notional)
            bumped = True
        else:
            reason = "SIZE_TOO_SMALL"
            return 0.0, "INVALID", reason, _details(
                input_type=input_type,
                notional_raw=notional_raw,
                notional_capped=notional_capped,
                max_notional=max_notional,
                min_notional=min_notional,
                bumped=False,
                cap_binding=bool(max_notional < min_notional),
            )

    notional_rounded = float(round(float(notional_capped), 6))
    if notional_rounded <= 0:
        reason = "SIZE_ZERO"
        return 0.0, "INVALID", reason, _details(
            input_type=input_type,
            notional_raw=notional_raw,
            notional_capped=notional_capped,
            max_notional=max_notional,
            min_notional=min_notional,
            bumped=bumped,
        )

    return notional_rounded, "VALID", "", _details(
        input_type=input_type,
        notional_raw=notional_raw,
        notional_capped=notional_capped,
        max_notional=max_notional,
        min_notional=min_notional,
        bumped=bumped,
    )


def _details(
    *,
    input_type: str,
    notional_raw: float,
    notional_capped: float,
    max_notional: float,
    min_notional: float,
    bumped: bool,
    cap_binding: bool = False,
) -> dict[str, Any]:
    return {
        "input_type": str(input_type),
        "notional_raw": float(notional_raw),
        "notional_capped": float(notional_capped),
        "max_notional": float(max_notional),
        "min_notional": float(min_notional),
        "bumped_to_min_notional": bool(bumped),
        "cap_binding": bool(cap_binding),
    }


def _resolve_stage24_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    if "evaluation" in cfg and isinstance(cfg.get("evaluation"), dict):
        return dict((cfg.get("evaluation", {}) or {}).get("stage24", {}))
    return dict(cfg)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def is_known_reject_reason(reason: str) -> bool:
    return str(reason) in set(EXECUTION_REJECT_REASONS)
