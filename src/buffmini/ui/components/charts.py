"""Matplotlib chart helpers for Stage-5 Results Studio."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_leverage_frontier(frontier: pd.DataFrame) -> plt.Figure | None:
    """Plot leverage vs drawdown and leverage vs ruin probability."""

    if frontier is None or frontier.empty:
        return None
    if not {"leverage", "maxdd_p95", "p_ruin", "method"}.issubset(set(frontier.columns)):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, frame in frontier.groupby("method"):
        ordered = frame.sort_values("leverage")
        axes[0].plot(ordered["leverage"], ordered["maxdd_p95"], marker="o", label=str(method))
        axes[1].plot(ordered["leverage"], ordered["p_ruin"], marker="o", label=str(method))
    axes[0].set_title("Leverage vs MaxDD p95")
    axes[0].set_xlabel("Leverage")
    axes[0].set_ylabel("MaxDD p95")
    axes[1].set_title("Leverage vs P(ruin)")
    axes[1].set_xlabel("Leverage")
    axes[1].set_ylabel("P(ruin)")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_selector_log_growth(selector_table: pd.DataFrame) -> plt.Figure | None:
    """Plot expected log-growth curve across leverage levels."""

    if selector_table is None or selector_table.empty:
        return None
    required = {"leverage", "expected_log_growth", "method"}
    if not required.issubset(set(selector_table.columns)):
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    for method, frame in selector_table.groupby("method"):
        ordered = frame.sort_values("leverage")
        ax.plot(ordered["leverage"], ordered["expected_log_growth"], marker="o", label=str(method))
    ax.set_title("Leverage vs Expected Log Growth")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Expected Log Growth")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_mc_quantiles(mc_quantiles: pd.DataFrame) -> plt.Figure | None:
    """Plot Monte Carlo return quantiles by method if available."""

    if mc_quantiles is None or mc_quantiles.empty:
        return None
    required = {"method", "metric", "quantile", "value"}
    if not required.issubset(set(mc_quantiles.columns)):
        return None
    subset = mc_quantiles[mc_quantiles["metric"] == "return_pct"].copy()
    if subset.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = subset.pivot_table(index="method", columns="quantile", values="value", aggfunc="mean")
    for method in pivot.index:
        y_values = [pivot.loc[method].get("p05", float("nan")), pivot.loc[method].get("median", float("nan")), pivot.loc[method].get("p95", float("nan"))]
        ax.plot(["p05", "median", "p95"], y_values, marker="o", label=str(method))
    ax.set_title("Monte Carlo Return Quantiles")
    ax.set_ylabel("Return pct")
    ax.legend()
    fig.tight_layout()
    return fig

