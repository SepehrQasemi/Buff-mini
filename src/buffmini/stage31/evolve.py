"""Deterministic evolutionary strategy synthesis for Stage-31."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buffmini.stage31.dsl import DSLStrategy, evaluate_strategy, explain_expr
from buffmini.utils.hashing import stable_hash


@dataclass(frozen=True)
class EvolverConfig:
    population_size: int = 120
    generations: int = 20
    elite_count: int = 20
    mutation_rate: float = 0.35
    crossover_rate: float = 0.45
    novelty_similarity_max: float = 0.90
    seed: int = 42
    max_depth: int = 4


def evolve_strategies(
    *,
    frame: pd.DataFrame,
    features: list[str],
    cfg: EvolverConfig | None = None,
) -> pd.DataFrame:
    conf = cfg or EvolverConfig()
    rng = np.random.default_rng(int(conf.seed))
    feats = [str(f) for f in features if str(f) in frame.columns]
    if not feats:
        raise ValueError("No usable features for evolution")

    population = [_random_strategy(rng, feats, max_depth=int(conf.max_depth), idx=i) for i in range(int(conf.population_size))]
    for gen in range(int(max(1, conf.generations))):
        scored = _score_population(frame, population)
        elites = _select_diverse_elites(scored, int(conf.elite_count), float(conf.novelty_similarity_max))
        next_pop = list(elites["strategy"].tolist())
        while len(next_pop) < int(conf.population_size):
            roll = float(rng.random())
            if roll < float(conf.crossover_rate) and len(elites) >= 2:
                a = elites.iloc[int(rng.integers(0, len(elites)))]["strategy"]
                b = elites.iloc[int(rng.integers(0, len(elites)))]["strategy"]
                child = crossover_strategy(a, b, rng=rng, idx=len(next_pop), generation=gen + 1)
            elif roll < float(conf.crossover_rate + conf.mutation_rate) and len(elites) >= 1:
                parent = elites.iloc[int(rng.integers(0, len(elites)))]["strategy"]
                child = mutate_strategy(parent, rng=rng, features=feats, generation=gen + 1, idx=len(next_pop), max_depth=int(conf.max_depth))
            else:
                child = _random_strategy(rng, feats, max_depth=int(conf.max_depth), idx=len(next_pop), generation=gen + 1)
            next_pop.append(child)
        population = next_pop

    final_scored = _score_population(frame, population)
    final_elites = _select_diverse_elites(final_scored, int(conf.elite_count), float(conf.novelty_similarity_max)).copy()
    final_elites = final_elites.sort_values(["fitness", "strategy_id"], ascending=[False, True]).reset_index(drop=True)
    return final_elites


def mutate_strategy(
    strategy: DSLStrategy,
    *,
    rng: np.random.Generator,
    features: list[str],
    generation: int,
    idx: int,
    max_depth: int,
) -> DSLStrategy:
    mutated_long = _mutate_expr(strategy.long_expr, rng=rng, features=features, depth=0, max_depth=max_depth)
    mutated_short = _mutate_expr(strategy.short_expr, rng=rng, features=features, depth=0, max_depth=max_depth)
    return DSLStrategy(
        name=f"mut_g{generation}_{idx}",
        long_expr=mutated_long,
        short_expr=mutated_short,
        exit_mode=str(strategy.exit_mode),
        stop_atr_multiple=float(strategy.stop_atr_multiple),
        take_profit_atr_multiple=float(strategy.take_profit_atr_multiple),
        max_hold_bars=int(strategy.max_hold_bars),
    )


def crossover_strategy(
    left: DSLStrategy,
    right: DSLStrategy,
    *,
    rng: np.random.Generator,
    generation: int,
    idx: int,
) -> DSLStrategy:
    _ = rng  # reserved for future stochastic subtree sampling
    return DSLStrategy(
        name=f"xov_g{generation}_{idx}",
        long_expr=dict(left.long_expr),
        short_expr=dict(right.short_expr),
        exit_mode=str(left.exit_mode),
        stop_atr_multiple=float(left.stop_atr_multiple),
        take_profit_atr_multiple=float(left.take_profit_atr_multiple),
        max_hold_bars=int(left.max_hold_bars),
    )


def signal_similarity(a: pd.Series, b: pd.Series) -> float:
    va = pd.to_numeric(a, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    vb = pd.to_numeric(b, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = int(min(len(va), len(vb)))
    if n <= 1:
        return 0.0
    va = va[:n]
    vb = vb[:n]
    std_a = float(np.std(va))
    std_b = float(np.std(vb))
    if std_a <= 1e-12 or std_b <= 1e-12:
        union = np.count_nonzero((va != 0.0) | (vb != 0.0))
        if union == 0:
            return 1.0
        overlap = np.count_nonzero((va != 0.0) & (vb != 0.0))
        return float(overlap / union)
    corr = float(np.corrcoef(va, vb)[0, 1])
    return float(abs(corr)) if np.isfinite(corr) else 0.0


def _score_population(frame: pd.DataFrame, population: list[DSLStrategy]) -> pd.DataFrame:
    close = pd.to_numeric(frame.get("close", 0.0), errors="coerce").fillna(0.0)
    ret = close.pct_change().fillna(0.0)
    rows: list[dict[str, Any]] = []
    for strategy in population:
        signal = evaluate_strategy(strategy, frame)
        shifted = signal.shift(1).fillna(0).astype(int)
        pnl = shifted * ret
        trade_count = int((shifted != 0).sum())
        pnl_mean = float(pnl.mean())
        pnl_std = float(pnl.std(ddof=0))
        sharpe_like = pnl_mean / (pnl_std + 1e-12)
        fitness = float(sharpe_like + 0.01 * np.log1p(trade_count))
        rows.append(
            {
                "strategy": strategy,
                "strategy_id": stable_hash(
                    {
                        "name": strategy.name,
                        "long": strategy.long_expr,
                        "short": strategy.short_expr,
                        "exit_mode": strategy.exit_mode,
                        "max_hold_bars": strategy.max_hold_bars,
                    },
                    length=16,
                ),
                "fitness": fitness,
                "trade_count": trade_count,
                "signal": signal,
                "explain_long": explain_expr(strategy.long_expr),
                "explain_short": explain_expr(strategy.short_expr),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(["fitness", "strategy_id"], ascending=[False, True]).reset_index(drop=True)
    return out


def _select_diverse_elites(scored: pd.DataFrame, elite_count: int, similarity_max: float) -> pd.DataFrame:
    if scored.empty:
        return scored
    keep_rows: list[pd.Series] = []
    for _, row in scored.iterrows():
        current = row["signal"]
        is_too_similar = False
        for existing in keep_rows:
            sim = signal_similarity(current, existing["signal"])
            if sim > float(similarity_max):
                is_too_similar = True
                break
        if not is_too_similar:
            keep_rows.append(row)
        if len(keep_rows) >= int(max(1, elite_count)):
            break
    if not keep_rows:
        keep_rows = [scored.iloc[0]]
    return pd.DataFrame(keep_rows).reset_index(drop=True)


def _random_strategy(
    rng: np.random.Generator,
    features: list[str],
    *,
    max_depth: int,
    idx: int,
    generation: int = 0,
) -> DSLStrategy:
    long_expr = _random_bool_expr(rng, features, depth=0, max_depth=max_depth)
    short_expr = _random_bool_expr(rng, features, depth=0, max_depth=max_depth)
    return DSLStrategy(
        name=f"rand_g{generation}_{idx}",
        long_expr=long_expr,
        short_expr=short_expr,
        exit_mode="fixed_atr",
        stop_atr_multiple=1.5,
        take_profit_atr_multiple=3.0,
        max_hold_bars=24,
    )


def _random_bool_expr(
    rng: np.random.Generator,
    features: list[str],
    *,
    depth: int,
    max_depth: int,
) -> dict[str, Any]:
    if depth >= max_depth - 1:
        return _random_comparator(rng, features, depth=depth, max_depth=max_depth)
    op = str(rng.choice(["and", "or", "cmp", "cross"]))
    if op in {"and", "or"}:
        return {
            "op": op,
            "left": _random_bool_expr(rng, features, depth=depth + 1, max_depth=max_depth),
            "right": _random_bool_expr(rng, features, depth=depth + 1, max_depth=max_depth),
        }
    if op == "cross":
        return {
            "op": "cross",
            "direction": str(rng.choice(["up", "down"])),
            "left": _random_numeric_expr(rng, features, depth=depth + 1, max_depth=max_depth),
            "right": _random_numeric_expr(rng, features, depth=depth + 1, max_depth=max_depth),
        }
    return _random_comparator(rng, features, depth=depth, max_depth=max_depth)


def _random_comparator(
    rng: np.random.Generator,
    features: list[str],
    *,
    depth: int,
    max_depth: int,
) -> dict[str, Any]:
    return {
        "op": str(rng.choice([">", "<"])),
        "left": _random_numeric_expr(rng, features, depth=depth + 1, max_depth=max_depth),
        "right": _random_numeric_expr(rng, features, depth=depth + 1, max_depth=max_depth),
    }


def _random_numeric_expr(
    rng: np.random.Generator,
    features: list[str],
    *,
    depth: int,
    max_depth: int,
) -> dict[str, Any]:
    if depth >= max_depth:
        return {"op": "feature", "name": str(rng.choice(features))}
    op = str(rng.choice(["feature", "rolling_mean", "rolling_std", "rank", "percentile", "const"]))
    if op == "feature":
        return {"op": "feature", "name": str(rng.choice(features))}
    if op == "const":
        return {"op": "const", "value": float(rng.uniform(-1.0, 1.0))}
    if op == "percentile":
        return {
            "op": "percentile",
            "x": _random_numeric_expr(rng, features, depth=depth + 1, max_depth=max_depth),
            "window": int(rng.integers(8, 96)),
            "q": float(rng.uniform(0.1, 0.9)),
        }
    return {
        "op": op,
        "x": _random_numeric_expr(rng, features, depth=depth + 1, max_depth=max_depth),
        "window": int(rng.integers(8, 96)),
    }


def _mutate_expr(
    expr: dict[str, Any],
    *,
    rng: np.random.Generator,
    features: list[str],
    depth: int,
    max_depth: int,
) -> dict[str, Any]:
    mutated = dict(expr)
    op = str(mutated.get("op", "")).lower()
    if op == "feature":
        if float(rng.random()) < 0.7:
            mutated["name"] = str(rng.choice(features))
        return mutated
    if op == "const":
        mutated["value"] = float(mutated.get("value", 0.0)) + float(rng.normal(0.0, 0.1))
        return mutated
    if "window" in mutated and float(rng.random()) < 0.7:
        mutated["window"] = int(max(2, int(mutated.get("window", 20)) + int(rng.integers(-6, 7))))
    if op == "percentile" and float(rng.random()) < 0.7:
        mutated["q"] = float(min(0.95, max(0.05, float(mutated.get("q", 0.5)) + float(rng.normal(0.0, 0.05)))))
    if "x" in mutated and isinstance(mutated["x"], dict):
        mutated["x"] = _mutate_expr(mutated["x"], rng=rng, features=features, depth=depth + 1, max_depth=max_depth)
    if "left" in mutated and isinstance(mutated["left"], dict):
        mutated["left"] = _mutate_expr(mutated["left"], rng=rng, features=features, depth=depth + 1, max_depth=max_depth)
    if "right" in mutated and isinstance(mutated["right"], dict):
        mutated["right"] = _mutate_expr(mutated["right"], rng=rng, features=features, depth=depth + 1, max_depth=max_depth)
    if depth < max_depth - 1 and float(rng.random()) < 0.12:
        return _random_numeric_expr(rng, features, depth=depth, max_depth=max_depth) if op in {"feature", "const", "rolling_mean", "rolling_std", "rank", "percentile"} else _random_bool_expr(rng, features, depth=depth, max_depth=max_depth)
    return mutated

