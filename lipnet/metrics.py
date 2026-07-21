from __future__ import annotations

import numpy as np


def _validate_binary_inputs(scores: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if s.shape[0] != y.shape[0]:
        raise ValueError(f"scores and y_true must match length, got {s.shape[0]} vs {y.shape[0]}")
    if s.size == 0:
        raise ValueError("scores is empty")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must be binary with values in {0,1}")
    return s, y


def roc_auc(scores: np.ndarray, y_true: np.ndarray) -> float:
    s, y = _validate_binary_inputs(scores, y_true)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    sum_pos = float(ranks[y == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc(scores: np.ndarray, y_true: np.ndarray) -> float:
    s, y = _validate_binary_inputs(scores, y_true)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    precision = np.concatenate(([1.0], precision.astype(np.float64)))
    recall = np.concatenate(([0.0], recall.astype(np.float64)))
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return float(ap)
