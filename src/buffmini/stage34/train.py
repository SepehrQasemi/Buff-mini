"""Stage-34 deterministic CPU model training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.utils.hashing import stable_hash


def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    models: tuple[str, ...] = ("logreg", "hgbt", "rf")
    calibration: str = "platt"
    train_frac: float = 0.70
    val_frac: float = 0.15
    logreg_steps: int = 300
    logreg_lr: float = 0.05
    logreg_l2: float = 0.005
    hgbt_estimators: int = 16
    rf_estimators: int = 32


def train_stage34_models(
    dataset: pd.DataFrame,
    *,
    feature_columns: list[str],
    cfg: TrainConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if dataset.empty:
        raise ValueError("dataset is empty")
    work = dataset.copy().sort_values("timestamp").reset_index(drop=True)
    feats = [str(c) for c in feature_columns if str(c) in work.columns]
    if not feats:
        raise ValueError("feature_columns empty after filtering")
    X = np.asarray(work.loc[:, feats].to_numpy(dtype=float), dtype=float)
    y_raw = pd.to_numeric(work.get("label_primary"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = (y_raw > 0.0).astype(float)

    n = int(X.shape[0])
    train_end = int(max(2, min(n - 2, round(n * float(cfg.train_frac)))))
    val_end = int(max(train_end + 1, min(n - 1, round(n * float(cfg.train_frac + cfg.val_frac)))))
    idx_train = np.arange(0, train_end)
    idx_val = np.arange(train_end, val_end)
    idx_test = np.arange(val_end, n)
    if idx_test.size == 0:
        idx_test = idx_val

    X_train = X[idx_train]
    X_val = X[idx_val]
    X_test = X[idx_test]
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    mu = np.nanmean(X_train, axis=0)
    sigma = np.nanstd(X_train, axis=0)
    sigma = np.where(sigma <= 1e-8, 1.0, sigma)
    X_train_z = (X_train - mu) / sigma
    X_val_z = (X_val - mu) / sigma
    X_test_z = (X_test - mu) / sigma

    rng = np.random.default_rng(int(cfg.seed))
    models: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for name in [str(v) for v in cfg.models]:
        if name == "logreg":
            model = _train_logreg(X_train_z, y_train, cfg=cfg)
            raw_val = _predict_logreg(model, X_val_z)
            raw_test = _predict_logreg(model, X_test_z)
        elif name == "hgbt":
            model = _train_boosted_stumps(X_train_z, y_train, seed=rng.integers(0, 2**31 - 1), n_estimators=int(cfg.hgbt_estimators))
            raw_val = _predict_stump_ensemble(model, X_val_z)
            raw_test = _predict_stump_ensemble(model, X_test_z)
        elif name == "rf":
            model = _train_bagged_stumps(X_train_z, y_train, seed=rng.integers(0, 2**31 - 1), n_estimators=int(cfg.rf_estimators))
            raw_val = _predict_stump_ensemble(model, X_val_z)
            raw_test = _predict_stump_ensemble(model, X_test_z)
        else:
            continue

        calibrated, calibrator = calibrate_probabilities(
            raw_prob=raw_val,
            y_true=y_val,
            method=str(cfg.calibration),
            seed=int(cfg.seed),
        )
        test_prob = apply_calibration(raw_test, calibrator)
        test_prob = np.clip(test_prob, 1e-6, 1.0 - 1e-6)
        summary = {
            "model_name": name,
            "train_rows": int(idx_train.size),
            "val_rows": int(idx_val.size),
            "test_rows": int(idx_test.size),
            "val_logloss": float(_log_loss(y_val, calibrated)),
            "test_logloss": float(_log_loss(y_test, test_prob)),
            "val_brier": float(_brier(y_val, calibrated)),
            "test_brier": float(_brier(y_test, test_prob)),
            "prob_mean": float(np.mean(test_prob)),
            "prob_std": float(np.std(test_prob)),
        }
        models[name] = {
            "name": name,
            "features": feats,
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "base_model": model,
            "calibrator": calibrator,
            "summary": summary,
        }
        summaries.append(summary)

    out_summary = {
        "rows_total": int(n),
        "splits": {"train": int(idx_train.size), "val": int(idx_val.size), "test": int(idx_test.size)},
        "feature_count": int(len(feats)),
        "models": summaries,
        "train_summary_hash": stable_hash(
            {
                "seed": int(cfg.seed),
                "models": summaries,
                "splits": {"train": int(idx_train.size), "val": int(idx_val.size), "test": int(idx_test.size)},
                "feature_count": int(len(feats)),
            },
            length=16,
        ),
    }
    return models, out_summary


def predict_model_proba(model: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    feats = [str(v) for v in model.get("features", [])]
    X = np.asarray(frame.loc[:, feats].to_numpy(dtype=float), dtype=float)
    mu = np.asarray(model.get("mu", []), dtype=float)
    sigma = np.asarray(model.get("sigma", []), dtype=float)
    sigma = np.where(sigma <= 1e-8, 1.0, sigma)
    Xz = (X - mu) / sigma
    base = dict(model.get("base_model", {}))
    kind = str(base.get("kind", ""))
    if kind == "logreg":
        raw = _predict_logreg(base, Xz)
    else:
        raw = _predict_stump_ensemble(base, Xz)
    calibrated = apply_calibration(raw, dict(model.get("calibrator", {})))
    return np.clip(calibrated, 1e-6, 1.0 - 1e-6)


def save_models(models: dict[str, dict[str, Any]], *, out_dir: Path) -> dict[str, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, model in sorted(models.items()):
        path = target / f"{name}.json"
        path.write_text(json.dumps(model, indent=2, allow_nan=False), encoding="utf-8")
        paths[name] = path
    return paths


def load_models(*, models_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(models_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[str(payload.get("name", path.stem))] = payload
    return out


def calibrate_probabilities(
    *,
    raw_prob: np.ndarray,
    y_true: np.ndarray,
    method: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    p = np.clip(np.asarray(raw_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y_true, dtype=float)
    mode = str(method).strip().lower()
    if mode == "none":
        return p, {"method": "none"}
    if mode == "isotonic":
        bins = _fit_isotonic(p, y)
        return _apply_isotonic(p, bins), {"method": "isotonic", "bins": bins}
    # default platt
    x = np.log(p / (1.0 - p))
    a = 1.0
    b = 0.0
    lr = 0.05
    l2 = 1e-3
    _ = seed  # deterministic, no randomness currently required
    for _step in range(250):
        pred = sigmoid(a * x + b)
        err = pred - y
        da = float(np.mean(err * x) + l2 * a)
        db = float(np.mean(err))
        a -= lr * da
        b -= lr * db
    calibrated = sigmoid(a * x + b)
    return calibrated, {"method": "platt", "a": float(a), "b": float(b)}


def apply_calibration(raw_prob: np.ndarray, calibrator: dict[str, Any]) -> np.ndarray:
    p = np.clip(np.asarray(raw_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    method = str(calibrator.get("method", "none")).lower()
    if method == "none":
        return p
    if method == "isotonic":
        return _apply_isotonic(p, list(calibrator.get("bins", [])))
    a = float(calibrator.get("a", 1.0))
    b = float(calibrator.get("b", 0.0))
    x = np.log(p / (1.0 - p))
    return sigmoid(a * x + b)


def _train_logreg(X: np.ndarray, y: np.ndarray, *, cfg: TrainConfig) -> dict[str, Any]:
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(int(max(10, cfg.logreg_steps))):
        p = sigmoid(X @ w + b)
        err = p - y
        grad_w = (X.T @ err) / max(1, n) + float(cfg.logreg_l2) * w
        grad_b = float(np.mean(err))
        w -= float(cfg.logreg_lr) * grad_w
        b -= float(cfg.logreg_lr) * grad_b
    return {"kind": "logreg", "w": w.tolist(), "b": float(b)}


def _predict_logreg(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    w = np.asarray(model.get("w", []), dtype=float)
    b = float(model.get("b", 0.0))
    if X.shape[1] != w.shape[0]:
        raise ValueError("feature dimension mismatch for logreg")
    return sigmoid(X @ w + b)


def _train_boosted_stumps(X: np.ndarray, y: np.ndarray, *, seed: int, n_estimators: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n, d = X.shape
    logits = np.zeros(n, dtype=float)
    stumps: list[dict[str, Any]] = []
    for _ in range(int(max(1, n_estimators))):
        p = sigmoid(logits)
        residual = y - p
        best_j = 0
        best_gain = -1.0
        best_thr = 0.0
        for j in range(d):
            col = X[:, j]
            thr = float(np.quantile(col, 0.5))
            left = residual[col <= thr]
            right = residual[col > thr]
            gain = abs(float(left.mean()) if left.size else 0.0) + abs(float(right.mean()) if right.size else 0.0)
            if gain > best_gain:
                best_gain = gain
                best_j = j
                best_thr = thr
        col = X[:, best_j]
        left_mask = col <= best_thr
        right_mask = ~left_mask
        left_val = float(np.mean(residual[left_mask])) if np.any(left_mask) else 0.0
        right_val = float(np.mean(residual[right_mask])) if np.any(right_mask) else 0.0
        lr = float(rng.uniform(0.2, 0.4))
        logits[left_mask] += lr * left_val
        logits[right_mask] += lr * right_val
        stumps.append({"feature": int(best_j), "threshold": float(best_thr), "left": float(lr * left_val), "right": float(lr * right_val)})
    return {"kind": "stump_ensemble", "mode": "boost", "stumps": stumps}


def _train_bagged_stumps(X: np.ndarray, y: np.ndarray, *, seed: int, n_estimators: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n, d = X.shape
    stumps: list[dict[str, Any]] = []
    for _ in range(int(max(1, n_estimators))):
        idx = rng.integers(0, n, size=n)
        j = int(rng.integers(0, d))
        col = X[idx, j]
        thr = float(np.quantile(col, float(rng.uniform(0.25, 0.75))))
        yb = y[idx]
        left_mask = col <= thr
        right_mask = ~left_mask
        left_rate = float(np.mean(yb[left_mask])) if np.any(left_mask) else float(np.mean(yb))
        right_rate = float(np.mean(yb[right_mask])) if np.any(right_mask) else float(np.mean(yb))
        left_logit = float(np.log(np.clip(left_rate, 1e-4, 1 - 1e-4) / np.clip(1.0 - left_rate, 1e-4, 1 - 1e-4)))
        right_logit = float(np.log(np.clip(right_rate, 1e-4, 1 - 1e-4) / np.clip(1.0 - right_rate, 1e-4, 1 - 1e-4)))
        stumps.append({"feature": int(j), "threshold": float(thr), "left": left_logit, "right": right_logit})
    return {"kind": "stump_ensemble", "mode": "bag", "stumps": stumps}


def _predict_stump_ensemble(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    stumps = list(model.get("stumps", []))
    if not stumps:
        return np.full(X.shape[0], 0.5, dtype=float)
    logits = np.zeros(X.shape[0], dtype=float)
    for stump in stumps:
        j = int(stump.get("feature", 0))
        thr = float(stump.get("threshold", 0.0))
        left = float(stump.get("left", 0.0))
        right = float(stump.get("right", 0.0))
        col = X[:, j]
        logits += np.where(col <= thr, left, right)
    logits /= float(len(stumps))
    return sigmoid(logits)


def _fit_isotonic(prob: np.ndarray, y: np.ndarray) -> list[list[float]]:
    order = np.argsort(prob)
    p_sorted = prob[order]
    y_sorted = y[order]
    bins: list[list[float]] = []
    for p, t in zip(p_sorted, y_sorted, strict=False):
        bins.append([float(p), float(p), float(t), 1.0])  # lo, hi, avg, count
        while len(bins) >= 2 and bins[-2][2] > bins[-1][2]:
            b2 = bins.pop()
            b1 = bins.pop()
            count = b1[3] + b2[3]
            avg = (b1[2] * b1[3] + b2[2] * b2[3]) / count
            bins.append([b1[0], b2[1], float(avg), float(count)])
    return [[float(lo), float(hi), float(avg)] for lo, hi, avg, _cnt in bins]


def _apply_isotonic(prob: np.ndarray, bins: list[list[float]]) -> np.ndarray:
    if not bins:
        return np.clip(prob, 1e-6, 1.0 - 1e-6)
    p = np.asarray(prob, dtype=float)
    out = np.zeros_like(p, dtype=float)
    for i, v in enumerate(p):
        assigned = False
        for lo, hi, avg in bins:
            if float(lo) <= float(v) <= float(hi):
                out[i] = float(avg)
                assigned = True
                break
        if not assigned:
            if v < bins[0][0]:
                out[i] = float(bins[0][2])
            else:
                out[i] = float(bins[-1][2])
    return np.clip(out, 1e-6, 1.0 - 1e-6)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    yv = np.asarray(y, dtype=float)
    pv = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return float(-np.mean(yv * np.log(pv) + (1.0 - yv) * np.log(1.0 - pv)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    yv = np.asarray(y, dtype=float)
    pv = np.asarray(p, dtype=float)
    return float(np.mean((yv - pv) ** 2))
