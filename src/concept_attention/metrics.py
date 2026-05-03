from __future__ import annotations

import numpy as np


def binary_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC via the Mann-Whitney rank statistic."""
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    pos = y_true == 1
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = ranks[pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_aupr_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision for binary labels."""
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = 0
    ap = 0.0
    for rank, yi in enumerate(y_sorted, start=1):
        if yi == 1:
            tp += 1
            ap += tp / rank
    return float(ap / n_pos)


def compute_auc_aupr(task: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    if task == "binary1":
        return {"AUROC": binary_auc_roc(y_true, y_prob), "AUPR": binary_aupr_ap(y_true, y_prob)}

    if task == "multiclass":
        k_classes = y_prob.shape[1]
        aucs, aps = [], []
        for k in range(k_classes):
            yt = (y_true == k).astype(np.int64)
            ys = y_prob[:, k]
            aucs.append(binary_auc_roc(yt, ys))
            aps.append(binary_aupr_ap(yt, ys))
        return {
            "AUROC_macro_ovr": float(np.nanmean(aucs)),
            "AUPR_macro_ovr": float(np.nanmean(aps)),
        }

    if task == "multilabel":
        k_labels = y_true.shape[1]
        aucs, aps = [], []
        for k in range(k_labels):
            aucs.append(binary_auc_roc(y_true[:, k].astype(np.int64), y_prob[:, k]))
            aps.append(binary_aupr_ap(y_true[:, k].astype(np.int64), y_prob[:, k]))
        return {"AUROC_macro": float(np.nanmean(aucs)), "AUPR_macro": float(np.nanmean(aps))}

    raise ValueError(f"Unknown metric task: {task}")
