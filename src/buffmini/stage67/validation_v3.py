"""Stage-67 time-series validation protocol v3."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.utils.hashing import stable_hash


def build_anchored_splits(
    *,
    n_rows: int,
    n_splits: int = 5,
    min_train: int = 128,
    test_size: int = 64,
    purge_gap: int = 8,
) -> list[dict[str, int]]:
    if n_rows <= min_train + test_size:
        return []
    splits: list[dict[str, int]] = []
    start_train = int(min_train)
    while len(splits) < int(max(1, n_splits)):
        test_start = start_train + int(purge_gap)
        test_end = test_start + int(test_size)
        if test_end > n_rows:
            break
        splits.append(
            {
                "train_start": 0,
                "train_end": int(start_train),
                "test_start": int(test_start),
                "test_end": int(test_end),
            }
        )
        start_train += int(test_size)
    return splits


def evaluate_validation_protocol_v3(
    *,
    dataset: pd.DataFrame,
    score_column: str,
    label_column: str,
    stage_a_survivors: int,
    stage_b_survivors: int,
    min_forward_trades: int = 10,
    min_forward_exposure: float = 0.01,
    min_median_forward_exp_lcb: float = 0.0,
    n_splits: int = 5,
    min_train_rows: int = 128,
    test_size_rows: int = 64,
    purge_gap_rows: int = 8,
) -> dict[str, Any]:
    if dataset.empty:
        return {
            "status": "PARTIAL",
            "split_count": 0,
            "usable_windows": 0,
            "mean_score": 0.0,
            "mean_label": 0.0,
            "median_forward_exp_lcb": 0.0,
            "gates_effective": False,
            "blocker_reason": "empty_dataset",
            "windows": [],
        }
    work = dataset.copy()
    work[score_column] = pd.to_numeric(work.get(score_column), errors="coerce").fillna(0.0)
    work[label_column] = pd.to_numeric(work.get(label_column), errors="coerce").fillna(0.0)
    splits = build_anchored_splits(
        n_rows=len(work),
        n_splits=int(max(1, n_splits)),
        min_train=int(max(16, min_train_rows)),
        test_size=int(max(8, test_size_rows)),
        purge_gap=int(max(0, purge_gap_rows)),
    )
    if not splits:
        return {
            "status": "PARTIAL",
            "split_count": 0,
            "usable_windows": 0,
            "mean_score": float(work[score_column].mean()),
            "mean_label": float(work[label_column].mean()),
            "median_forward_exp_lcb": float(work[score_column].median()),
            "gates_effective": False,
            "blocker_reason": "insufficient_rows_for_splits",
            "windows": [],
        }
    per_fold_scores: list[float] = []
    per_fold_labels: list[float] = []
    window_rows: list[dict[str, Any]] = []
    usable_scores: list[float] = []
    for split in splits:
        fold = work.iloc[split["test_start"] : split["test_end"]]
        fold_score = float(fold[score_column].mean())
        fold_label = float(fold[label_column].mean())
        fold_trade_count = int((fold[label_column] > 0.0).sum())
        exposure_col = "realized_label_present" if "realized_label_present" in fold.columns else label_column
        fold_exposure = float(pd.to_numeric(fold.get(exposure_col, 0.0), errors="coerce").fillna(0.0).mean())
        usable = bool(
            fold_trade_count >= int(max(1, min_forward_trades))
            and fold_exposure >= float(max(0.0, min_forward_exposure))
        )
        per_fold_scores.append(fold_score)
        per_fold_labels.append(fold_label)
        if usable:
            usable_scores.append(fold_score)
        window_rows.append(
            {
                "train_start": int(split["train_start"]),
                "train_end": int(split["train_end"]),
                "test_start": int(split["test_start"]),
                "test_end": int(split["test_end"]),
                "forward_trade_count": int(fold_trade_count),
                "forward_exposure": float(round(fold_exposure, 8)),
                "forward_exp_lcb": float(round(fold_score, 8)),
                "forward_label_mean": float(round(fold_label, 8)),
                "usable": bool(usable),
            }
        )
    mean_score = float(np.mean(per_fold_scores)) if per_fold_scores else 0.0
    mean_label = float(np.mean(per_fold_labels)) if per_fold_labels else 0.0
    usable_windows = int(len(usable_scores))
    median_forward_exp_lcb = float(np.median(usable_scores)) if usable_scores else float(-1.0)
    gates_effective = bool(int(stage_a_survivors) > 0 and int(stage_b_survivors) > 0)
    status = "SUCCESS" if gates_effective and mean_label > 0.0 and median_forward_exp_lcb >= float(min_median_forward_exp_lcb) and usable_windows > 0 else "PARTIAL"
    blocker = "" if status == "SUCCESS" else "validation_or_survivor_gate_not_met"
    payload = {
        "status": status,
        "split_count": int(len(splits)),
        "usable_windows": int(usable_windows),
        "mean_score": float(round(mean_score, 8)),
        "mean_label": float(round(mean_label, 8)),
        "median_forward_exp_lcb": float(round(median_forward_exp_lcb, 8)),
        "gates_effective": gates_effective,
        "blocker_reason": blocker,
        "windows": window_rows,
    }
    payload["summary_hash"] = stable_hash(payload, length=16)
    return payload
