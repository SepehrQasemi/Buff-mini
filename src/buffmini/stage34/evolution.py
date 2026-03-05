"""Stage-34 deterministic self-improving evolution engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.stage34.eval import EvalConfig, evaluate_models_strict
from buffmini.stage34.model_registry import RegistryEntry, select_elites, upsert_entry
from buffmini.stage34.train import TrainConfig, train_stage34_models
from buffmini.utils.hashing import stable_hash


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int = 10
    max_models_per_generation: int = 12
    exploration_pct: float = 0.20
    seed: int = 42
    budget: str = "small"


def run_evolution(
    dataset: pd.DataFrame,
    *,
    feature_pool: list[str],
    registry_path: Path,
    cfg: EvolutionConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(cfg.seed))
    generations = int(max(1, cfg.generations))
    per_gen = int(max(2, cfg.max_models_per_generation))
    exploration_n = int(max(1, np.ceil(per_gen * max(float(cfg.exploration_pct), 0.20))))
    summaries: list[dict[str, Any]] = []

    prev_population: list[dict[str, Any]] = []
    for gen in range(generations):
        elites = select_elites(Path(registry_path), generation=max(0, gen - 1), elite_count=max(1, per_gen - exploration_n))
        population = _build_population(
            generation=gen,
            feature_pool=feature_pool,
            per_gen=per_gen,
            exploration_n=exploration_n,
            rng=rng,
            elites=elites,
            prev_population=prev_population,
        )
        evaluated: list[dict[str, Any]] = []
        for idx, candidate in enumerate(population):
            entry = _evaluate_candidate(dataset, candidate=candidate, seed=int(cfg.seed + gen * 100 + idx))
            evaluated.append(entry)
            reg = RegistryEntry(
                model_id=str(entry["model_id"]),
                generation=int(gen),
                seed=int(cfg.seed),
                symbol=str(entry.get("symbol", "BTC/USDT")),
                timeframe=str(entry.get("timeframe", "1h")),
                horizon=str(entry.get("horizon", "24h")),
                feature_subset_sig=str(entry.get("feature_subset_sig", "")),
                hyperparameters=dict(entry.get("hyperparameters", {})),
                metrics=dict(entry.get("metrics", {})),
                data_hash=str(entry.get("data_hash", "")),
                resolved_end_ts=str(entry.get("resolved_end_ts", None)),
                parent_model_ids=tuple(str(v) for v in entry.get("parent_model_ids", [])),
            )
            upsert_entry(Path(registry_path), reg)

        ranked = sorted(evaluated, key=lambda row: (float(row["metrics"]["exp_lcb"]), float(row["metrics"]["positive_windows_ratio"]), -float(row["metrics"]["maxdd_p95"])), reverse=True)
        best = ranked[0]
        median = ranked[len(ranked) // 2]
        worst = ranked[-1]
        no_op = _is_generation_no_op(ranked)
        if no_op:
            raise RuntimeError(f"Generation {gen} is no-op: no diversity in configs/features")
        summary = {
            "generation": int(gen),
            "candidate_count": int(len(ranked)),
            "exploration_count": int(sum(1 for row in ranked if bool(row.get("exploratory", False)))),
            "best": _summary_view(best),
            "median": _summary_view(median),
            "worst": _summary_view(worst),
            "failure_mode_counts": _failure_counts(ranked),
            "improved_vs_prev_best": bool(
                not summaries or float(best["metrics"]["exp_lcb"]) > float(summaries[-1]["best"]["exp_lcb"])
            ),
        }
        summaries.append(summary)
        prev_population = population

    return {
        "generations": summaries,
        "best_generation": int(np.argmax([float(g["best"]["exp_lcb"]) for g in summaries])),
        "did_generations_improve": bool(any(bool(g.get("improved_vs_prev_best", False)) for g in summaries[1:])),
    }


def _evaluate_candidate(dataset: pd.DataFrame, *, candidate: dict[str, Any], seed: int) -> dict[str, Any]:
    feats = [str(v) for v in candidate.get("features", []) if str(v) in dataset.columns]
    if not feats:
        feats = [c for c in dataset.columns if c.startswith("ret_")][:6]
    model_name = str(candidate.get("model_name", "logreg"))
    threshold = float(candidate.get("threshold", 0.55))
    train_models, _train_summary = train_stage34_models(
        dataset,
        feature_columns=feats,
        cfg=TrainConfig(seed=int(seed), models=(model_name,), calibration="platt"),
    )
    eval_rows, eval_summary = evaluate_models_strict(
        dataset,
        models=train_models,
        cfg=EvalConfig(
            threshold=float(threshold),
            window_months=tuple(int(v) for v in candidate.get("window_months", (3,))),
            step_months=1,
            min_usable_windows=1,
            mc_min_trades=5,
            seed=int(seed),
        ),
    )
    live = eval_rows.loc[eval_rows["cost_mode"] == "live"].copy() if not eval_rows.empty else pd.DataFrame()
    row = live.sort_values("exp_lcb", ascending=False).head(1)
    if row.empty and not eval_rows.empty:
        row = eval_rows.sort_values("exp_lcb", ascending=False).head(1)
    metrics = {
        "exp_lcb": float(row["exp_lcb"].iloc[0]) if not row.empty else 0.0,
        "positive_windows_ratio": float(row["positive_windows_ratio"].iloc[0]) if not row.empty else 0.0,
        "trade_count": int(row["trade_count"].iloc[0]) if not row.empty else 0,
        "maxdd_p95": float(row["maxdd_p95"].iloc[0]) if not row.empty else 1.0,
        "wf_executed_pct": float(eval_summary.get("wf_executed_pct", 0.0)),
        "mc_trigger_pct": float(eval_summary.get("mc_trigger_pct", 0.0)),
        "failure_mode": str(row["failure_mode"].iloc[0]) if not row.empty else "insufficient_sample",
    }
    model_id = f"m_{stable_hash({'seed': int(seed), 'model': model_name, 'features': feats, 'threshold': threshold, 'windows': candidate.get('window_months', (3,))}, length=16)}"
    return {
        "model_id": model_id,
        "generation": int(candidate.get("generation", 0)),
        "symbol": "BTC/USDT",
        "timeframe": str(candidate.get("timeframe", "1h")),
        "horizon": str(candidate.get("horizon", "24h")),
        "feature_subset_sig": stable_hash(feats, length=12),
        "hyperparameters": {
            "model_name": model_name,
            "threshold": float(threshold),
            "window_months": list(candidate.get("window_months", (3,))),
        },
        "metrics": metrics,
        "data_hash": stable_hash(dataset.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="list"), length=16),
        "resolved_end_ts": str(pd.to_datetime(dataset["timestamp"], utc=True, errors="coerce").max()),
        "parent_model_ids": list(candidate.get("parent_model_ids", [])),
        "exploratory": bool(candidate.get("exploratory", False)),
    }


def _build_population(
    *,
    generation: int,
    feature_pool: list[str],
    per_gen: int,
    exploration_n: int,
    rng: np.random.Generator,
    elites: list[dict[str, Any]],
    prev_population: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    models = ["logreg", "hgbt", "rf"]
    population: list[dict[str, Any]] = []
    # Exploratory candidates.
    for _ in range(exploration_n):
        population.append(
            {
                "generation": int(generation),
                "model_name": str(models[int(rng.integers(0, len(models)))]),
                "threshold": float(rng.uniform(0.50, 0.65)),
                "window_months": (3, 6) if float(rng.random()) < 0.5 else (3,),
                "features": _sample_features(rng=rng, pool=feature_pool, min_k=6, max_k=min(20, len(feature_pool))),
                "parent_model_ids": [],
                "exploratory": True,
            }
        )
    # Elite offspring.
    while len(population) < per_gen:
        if elites:
            parent = elites[int(rng.integers(0, len(elites)))]
            parent_hp = dict(parent.get("hyperparameters", {}))
            parent_features = _parent_features_sig_to_list(parent, feature_pool)
            mutated = _mutate_candidate(
                rng=rng,
                generation=generation,
                parent_model_id=str(parent.get("model_id", "")),
                parent_model_name=str(parent_hp.get("model_name", "logreg")),
                parent_threshold=float(parent_hp.get("threshold", 0.55)),
                parent_windows=tuple(int(v) for v in parent_hp.get("window_months", [3])),
                parent_features=parent_features,
                feature_pool=feature_pool,
            )
            population.append(mutated)
        else:
            population.append(
                {
                    "generation": int(generation),
                    "model_name": str(models[int(rng.integers(0, len(models)))]),
                    "threshold": float(rng.uniform(0.50, 0.65)),
                    "window_months": (3,),
                    "features": _sample_features(rng=rng, pool=feature_pool, min_k=6, max_k=min(20, len(feature_pool))),
                    "parent_model_ids": [],
                    "exploratory": False,
                }
            )
    # Deterministic sort by signature for stable order.
    population = sorted(
        population,
        key=lambda c: stable_hash(
            {
                "g": int(c.get("generation", 0)),
                "m": str(c.get("model_name", "")),
                "t": float(c.get("threshold", 0.0)),
                "w": list(c.get("window_months", ())),
                "f": list(c.get("features", [])),
                "p": list(c.get("parent_model_ids", [])),
                "e": bool(c.get("exploratory", False)),
            },
            length=20,
        ),
    )
    return population[:per_gen]


def _sample_features(*, rng: np.random.Generator, pool: list[str], min_k: int, max_k: int) -> list[str]:
    if not pool:
        return []
    k = int(rng.integers(max(1, min_k), max(min_k + 1, max_k + 1)))
    k = max(1, min(k, len(pool)))
    idx = rng.choice(len(pool), size=k, replace=False)
    return sorted([str(pool[int(i)]) for i in idx])


def _mutate_candidate(
    *,
    rng: np.random.Generator,
    generation: int,
    parent_model_id: str,
    parent_model_name: str,
    parent_threshold: float,
    parent_windows: tuple[int, ...],
    parent_features: list[str],
    feature_pool: list[str],
) -> dict[str, Any]:
    models = ["logreg", "hgbt", "rf"]
    model_name = parent_model_name if float(rng.random()) >= 0.25 else str(models[int(rng.integers(0, len(models)))])
    threshold = float(np.clip(parent_threshold + rng.normal(0.0, 0.03), 0.50, 0.70))
    if float(rng.random()) < 0.20:
        windows = (3, 6) if tuple(parent_windows) == (3,) else (3,)
    else:
        windows = tuple(parent_windows) if parent_windows else (3,)
    features = list(parent_features) if parent_features else _sample_features(rng=rng, pool=feature_pool, min_k=6, max_k=min(20, len(feature_pool)))
    if features and float(rng.random()) < 0.6:
        drop_n = int(rng.integers(1, max(2, min(3, len(features)))))
        for _ in range(drop_n):
            if len(features) <= 3:
                break
            features.pop(int(rng.integers(0, len(features))))
    add_pool = [f for f in feature_pool if f not in features]
    if add_pool and float(rng.random()) < 0.8:
        add_n = int(rng.integers(1, max(2, min(4, len(add_pool)))))
        add_idx = rng.choice(len(add_pool), size=min(add_n, len(add_pool)), replace=False)
        for idx in add_idx:
            features.append(str(add_pool[int(idx)]))
    features = sorted(set(features))[:20]
    return {
        "generation": int(generation),
        "model_name": str(model_name),
        "threshold": float(threshold),
        "window_months": windows,
        "features": features,
        "parent_model_ids": [str(parent_model_id)] if parent_model_id else [],
        "exploratory": False,
    }


def _parent_features_sig_to_list(parent: dict[str, Any], feature_pool: list[str]) -> list[str]:
    sig = str(parent.get("feature_subset_sig", ""))
    if not sig:
        return feature_pool[: min(10, len(feature_pool))]
    # deterministic pseudo-reconstruction when exact list is not persisted.
    out: list[str] = []
    for feat in sorted(feature_pool):
        h = stable_hash({"sig": sig, "feat": feat}, length=8)
        if int(h, 16) % 3 == 0:
            out.append(feat)
    if not out:
        out = feature_pool[: min(10, len(feature_pool))]
    return out[:20]


def _summary_view(row: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(row.get("metrics", {}))
    return {
        "model_id": str(row.get("model_id", "")),
        "exp_lcb": float(metrics.get("exp_lcb", 0.0)),
        "positive_windows_ratio": float(metrics.get("positive_windows_ratio", 0.0)),
        "trade_count": int(metrics.get("trade_count", 0)),
        "maxdd_p95": float(metrics.get("maxdd_p95", 1.0)),
        "failure_mode": str(metrics.get("failure_mode", "")),
    }


def _failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        mode = str((row.get("metrics", {}) or {}).get("failure_mode", "unknown"))
        out[mode] = int(out.get(mode, 0) + 1)
    return out


def _is_generation_no_op(rows: list[dict[str, Any]]) -> bool:
    sigs = set()
    for row in rows:
        hp = dict(row.get("hyperparameters", {}))
        sig = stable_hash({"m": hp.get("model_name", ""), "t": hp.get("threshold", 0.0), "w": hp.get("window_months", []), "f": row.get("feature_subset_sig", "")}, length=16)
        sigs.add(sig)
    return len(sigs) <= 1
