from __future__ import annotations

from typing import Iterable

import numpy as np


def mean_std(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std}


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    sorted_vals = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def binary_auc(scores: np.ndarray, positive_mask: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    n_pos = int(positive_mask.sum())
    n_neg = int((~positive_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    auc = (ranks[positive_mask].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def topk_stats(scores: np.ndarray, positive_mask: np.ndarray, k: int) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    total_pos = int(positive_mask.sum())
    k_eff = min(max(int(k), 1), len(scores))
    order = np.argsort(scores)[::-1][:k_eff]
    selected_pos = int(positive_mask[order].sum())
    return {
        "selected_positive": selected_pos,
        "purity": float(selected_pos / max(k_eff, 1)),
        "recall": float(selected_pos / max(total_pos, 1)),
    }


def spearman_rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    ra = average_ranks(a)
    rb = average_ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom <= 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)
