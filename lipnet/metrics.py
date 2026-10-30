from __future__ import annotations

import numpy as np

EVALUATOR_VERSION = "binary-ranking-v2-ties"


def _validate_binary_inputs(scores: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    raw_y = np.asarray(y_true).reshape(-1)
    if s.size != raw_y.size or not s.size:
        raise ValueError("scores and y_true must have the same nonzero length")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    # Validate before integer conversion: 0.5 must not silently become 0.
    if not np.isin(raw_y, [0, 1]).all():
        raise ValueError("y_true must be binary with values in {0,1}")
    return s, raw_y.astype(np.int64)


def roc_auc(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Mann–Whitney AUROC, assigning half credit to equal-score pairs."""
    s, y = _validate_binary_inputs(scores, y_true)
    n_pos = int(y.sum())
    n_neg = s.size - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ordered = s[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], s.size]
    positives = np.add.reduceat(y[order], starts)
    mean_ranks = (starts + 1 + ends) / 2.0
    return float((np.dot(positives, mean_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Non-interpolated average precision at distinct score thresholds."""
    s, y = _validate_binary_inputs(scores, y_true)
    n_pos = int(y.sum())
    if not n_pos:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    ordered = s[order]
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), s.size - 1]
    tp = np.cumsum(y[order])[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) / n_pos * tp / (ends + 1)))
