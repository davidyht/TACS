import numpy as np

from tacs.metrics import average_ranks, binary_auc, spearman_rank_correlation, topk_stats


def test_average_ranks_handles_ties():
    np.testing.assert_allclose(average_ranks(np.array([2.0, 2.0, 1.0])), [2.5, 2.5, 1.0])


def test_binary_auc_perfect_ranking():
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    positives = np.array([False, True, True, False])
    assert binary_auc(scores, positives) == 1.0


def test_topk_stats():
    stats = topk_stats(np.array([0.9, 0.1, 0.8]), np.array([True, False, True]), k=2)
    assert stats["selected_positive"] == 2
    assert stats["purity"] == 1.0
    assert stats["recall"] == 1.0


def test_spearman_rank_correlation_identity():
    x = np.array([3.0, 1.0, 2.0])
    assert spearman_rank_correlation(x, x) == 1.0
