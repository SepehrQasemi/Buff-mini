"""Stage-27.9 contextual evidence thresholding."""

from __future__ import annotations

from typing import Any

import pandas as pd


def classify_context_evidence(
    *,
    occurrences: int,
    trades: int,
    exp_lcb: float,
    positive_windows_ratio: float,
) -> str:
    """Classify contextual edge evidence with deterministic thresholds."""

    occ = int(max(0, occurrences))
    trd = int(max(0, trades))
    lcb = float(exp_lcb)
    pos_ratio = float(max(0.0, min(1.0, positive_windows_ratio)))
    if occ >= 50 and trd >= 30 and lcb > 0.0 and pos_ratio >= 0.55:
        return "ROBUST_CONTEXT_EDGE"
    if lcb > 0.0 and pos_ratio > 0.0 and (occ >= 25 or trd >= 15):
        return "WEAK_CONTEXT_EDGE"
    return "NOISE"


def evaluate_context_evidence(
    rolling_results: pd.DataFrame,
) -> dict[str, Any]:
    """Aggregate rolling contextual outcomes into evidence-based classifications."""

    if rolling_results.empty:
        return {
            "rows": [],
            "counts": {
                "ROBUST_CONTEXT_EDGE": 0,
                "WEAK_CONTEXT_EDGE": 0,
                "NOISE": 0,
            },
        }

    work = rolling_results.copy()
    for col, default in (
        ("context", ""),
        ("rulelet", ""),
        ("timeframe", ""),
        ("symbol", ""),
    ):
        if col not in work.columns:
            work[col] = default
        work[col] = work[col].astype(str)

    if "context_occurrences" not in work.columns:
        work["context_occurrences"] = 0
    if "trade_count" not in work.columns:
        work["trade_count"] = 0
    if "exp_lcb" not in work.columns:
        work["exp_lcb"] = 0.0

    work["context_occurrences"] = pd.to_numeric(work["context_occurrences"], errors="coerce").fillna(0).astype(int)
    work["trade_count"] = pd.to_numeric(work["trade_count"], errors="coerce").fillna(0).astype(int)
    work["exp_lcb"] = pd.to_numeric(work["exp_lcb"], errors="coerce").fillna(0.0)

    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(["symbol", "timeframe", "context", "rulelet"], dropna=False):
        symbol, timeframe, context, rulelet = keys
        windows_total = int(group.shape[0])
        windows_positive = int((group["exp_lcb"] > 0.0).sum())
        positive_ratio = float(windows_positive / max(1, windows_total))
        occ_med = int(group["context_occurrences"].median()) if windows_total > 0 else 0
        trades_med = int(group["trade_count"].median()) if windows_total > 0 else 0
        lcb_med = float(group["exp_lcb"].median()) if windows_total > 0 else 0.0
        verdict = classify_context_evidence(
            occurrences=occ_med,
            trades=trades_med,
            exp_lcb=lcb_med,
            positive_windows_ratio=positive_ratio,
        )
        rows.append(
            {
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "context": str(context),
                "rulelet": str(rulelet),
                "windows_total": int(windows_total),
                "windows_positive": int(windows_positive),
                "positive_windows_ratio": float(positive_ratio),
                "occurrences_median": int(occ_med),
                "trades_median": int(trades_med),
                "exp_lcb_median": float(lcb_med),
                "classification": str(verdict),
            }
        )

    rows_df = pd.DataFrame(rows)
    counts = {
        "ROBUST_CONTEXT_EDGE": int((rows_df["classification"] == "ROBUST_CONTEXT_EDGE").sum()) if not rows_df.empty else 0,
        "WEAK_CONTEXT_EDGE": int((rows_df["classification"] == "WEAK_CONTEXT_EDGE").sum()) if not rows_df.empty else 0,
        "NOISE": int((rows_df["classification"] == "NOISE").sum()) if not rows_df.empty else 0,
    }

    top_edges = []
    if not rows_df.empty:
        top_df = rows_df.sort_values(
            ["classification", "exp_lcb_median", "positive_windows_ratio", "trades_median"],
            ascending=[True, False, False, False],
        ).head(20)
        top_edges = top_df.to_dict(orient="records")

    return {
        "rows": rows,
        "counts": counts,
        "top_edges": top_edges,
    }


def render_context_evidence_md(summary: dict[str, Any]) -> str:
    """Render a short deterministic markdown report."""

    counts = dict(summary.get("counts", {}))
    lines = [
        "# Stage-27.9 Context Evidence",
        "",
        "## Classification Rules",
        "- ROBUST_CONTEXT_EDGE: occurrences >= 50, trades >= 30, exp_lcb > 0, positive_windows_ratio >= 0.55",
        "- WEAK_CONTEXT_EDGE: partial positive evidence with non-trivial support",
        "- NOISE: evidence below thresholds",
        "",
        "## Counts",
        f"- ROBUST_CONTEXT_EDGE: `{int(counts.get('ROBUST_CONTEXT_EDGE', 0))}`",
        f"- WEAK_CONTEXT_EDGE: `{int(counts.get('WEAK_CONTEXT_EDGE', 0))}`",
        f"- NOISE: `{int(counts.get('NOISE', 0))}`",
        "",
        "## Top Context Edges",
    ]
    top = list(summary.get("top_edges", []))
    if top:
        lines.append("| symbol | timeframe | context | rulelet | windows | positive_ratio | exp_lcb_median | trades_median | class |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---|")
        for row in top[:15]:
            lines.append(
                "| {symbol} | {timeframe} | {context} | {rulelet} | {windows_total} | {positive_windows_ratio:.3f} | {exp_lcb_median:.6f} | {trades_median} | {classification} |".format(
                    **{
                        "symbol": row.get("symbol", ""),
                        "timeframe": row.get("timeframe", ""),
                        "context": row.get("context", ""),
                        "rulelet": row.get("rulelet", ""),
                        "windows_total": int(row.get("windows_total", 0)),
                        "positive_windows_ratio": float(row.get("positive_windows_ratio", 0.0)),
                        "exp_lcb_median": float(row.get("exp_lcb_median", 0.0)),
                        "trades_median": int(row.get("trades_median", 0)),
                        "classification": row.get("classification", ""),
                    }
                )
            )
    else:
        lines.append("- No contextual rows were available.")
    return "\n".join(lines).strip() + "\n"
