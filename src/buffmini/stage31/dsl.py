"""Interpretable DSL for Stage-31 strategy synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


MAX_DSL_DEPTH = 4


@dataclass(frozen=True)
class DSLStrategy:
    name: str
    long_expr: dict[str, Any]
    short_expr: dict[str, Any]
    exit_mode: str = "fixed_atr"
    stop_atr_multiple: float = 1.5
    take_profit_atr_multiple: float = 3.0
    max_hold_bars: int = 24

    def explain(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "long_rule": explain_expr(self.long_expr),
            "short_rule": explain_expr(self.short_expr),
            "exit_mode": self.exit_mode,
            "stop_atr_multiple": float(self.stop_atr_multiple),
            "take_profit_atr_multiple": float(self.take_profit_atr_multiple),
            "max_hold_bars": int(self.max_hold_bars),
        }


def evaluate_strategy(strategy: DSLStrategy, frame: pd.DataFrame) -> pd.Series:
    long_mask = evaluate_bool_expr(strategy.long_expr, frame).fillna(False)
    short_mask = evaluate_bool_expr(strategy.short_expr, frame).fillna(False)
    signal = pd.Series(0, index=frame.index, dtype=int)
    signal.loc[long_mask & ~short_mask] = 1
    signal.loc[short_mask & ~long_mask] = -1
    return signal.shift(1).fillna(0).astype(int)


def evaluate_bool_expr(expr: dict[str, Any], frame: pd.DataFrame, *, depth: int = 0) -> pd.Series:
    _validate_depth(depth)
    op = str(expr.get("op", "")).strip().lower()
    if op in {"and", "or"}:
        left = evaluate_bool_expr(dict(expr.get("left", {})), frame, depth=depth + 1)
        right = evaluate_bool_expr(dict(expr.get("right", {})), frame, depth=depth + 1)
        return (left & right) if op == "and" else (left | right)
    if op in {">", "<"}:
        left_num = evaluate_numeric_expr(dict(expr.get("left", {})), frame, depth=depth + 1)
        right_num = evaluate_numeric_expr(dict(expr.get("right", {})), frame, depth=depth + 1)
        return (left_num > right_num) if op == ">" else (left_num < right_num)
    if op == "cross":
        left_num = evaluate_numeric_expr(dict(expr.get("left", {})), frame, depth=depth + 1)
        right_num = evaluate_numeric_expr(dict(expr.get("right", {})), frame, depth=depth + 1)
        direction = str(expr.get("direction", "up")).strip().lower()
        prev_diff = (left_num - right_num).shift(1)
        curr_diff = (left_num - right_num)
        if direction == "down":
            return (prev_diff >= 0.0) & (curr_diff < 0.0)
        return (prev_diff <= 0.0) & (curr_diff > 0.0)
    raise ValueError(f"Unsupported boolean DSL op: {op}")


def evaluate_numeric_expr(expr: dict[str, Any], frame: pd.DataFrame, *, depth: int = 0) -> pd.Series:
    _validate_depth(depth)
    op = str(expr.get("op", "")).strip().lower()
    if op == "feature":
        name = str(expr.get("name", "")).strip()
        if name not in frame.columns:
            return pd.Series(0.0, index=frame.index, dtype=float)
        return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
    if op == "const":
        return pd.Series(float(expr.get("value", 0.0)), index=frame.index, dtype=float)
    if op in {"rolling_mean", "rolling_std", "rank", "percentile"}:
        x = evaluate_numeric_expr(dict(expr.get("x", {})), frame, depth=depth + 1)
        window = int(max(2, int(expr.get("window", 20))))
        if op == "rolling_mean":
            return x.rolling(window=window, min_periods=2).mean().fillna(0.0)
        if op == "rolling_std":
            return x.rolling(window=window, min_periods=2).std(ddof=0).fillna(0.0)
        if op == "rank":
            return x.rolling(window=window, min_periods=2).apply(_last_rank, raw=True).fillna(0.0)
        q = float(expr.get("q", 0.5))
        q = float(min(1.0, max(0.0, q)))
        return x.rolling(window=window, min_periods=2).quantile(q).fillna(0.0)
    if op in {"add", "sub", "mul", "div"}:
        left = evaluate_numeric_expr(dict(expr.get("left", {})), frame, depth=depth + 1)
        right = evaluate_numeric_expr(dict(expr.get("right", {})), frame, depth=depth + 1)
        if op == "add":
            return left + right
        if op == "sub":
            return left - right
        if op == "mul":
            return left * right
        denom = right.replace(0.0, np.nan)
        return (left / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raise ValueError(f"Unsupported numeric DSL op: {op}")


def explain_expr(expr: dict[str, Any]) -> str:
    op = str(expr.get("op", "")).strip().lower()
    if op == "feature":
        return str(expr.get("name", "unknown_feature"))
    if op == "const":
        return f"{float(expr.get('value', 0.0)):.6f}"
    if op in {"rolling_mean", "rolling_std", "rank", "percentile"}:
        inner = explain_expr(dict(expr.get("x", {})))
        window = int(expr.get("window", 20))
        if op == "percentile":
            return f"percentile({inner}, w={window}, q={float(expr.get('q', 0.5)):.2f})"
        return f"{op}({inner}, w={window})"
    if op in {"add", "sub", "mul", "div", ">", "<", "and", "or"}:
        left = explain_expr(dict(expr.get("left", {})))
        right = explain_expr(dict(expr.get("right", {})))
        return f"({left} {op} {right})"
    if op == "cross":
        left = explain_expr(dict(expr.get("left", {})))
        right = explain_expr(dict(expr.get("right", {})))
        direction = str(expr.get("direction", "up")).strip().lower()
        return f"cross_{direction}({left}, {right})"
    return f"unknown({op})"


def _validate_depth(depth: int) -> None:
    if int(depth) > int(MAX_DSL_DEPTH):
        raise ValueError(f"DSL depth exceeded max depth {MAX_DSL_DEPTH}")


def _last_rank(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    x = float(arr[-1])
    less = float(np.sum(arr < x))
    equal = float(np.sum(arr == x))
    return float((less + 0.5 * equal) / max(1.0, float(arr.size)))

