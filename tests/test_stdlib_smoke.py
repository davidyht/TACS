import subprocess
import sys
import unittest

import numpy as np
import torch

from tacs.data_selection.matching import calculate_influence_score
from tacs.data_selection.select_from_scores import select_indices
from tacs.metrics import (
    average_ranks,
    binary_auc,
    final_drop_from_loss_stack,
    spearman_rank_correlation,
    tacs_from_loss_stack,
    topk_stats,
)


class ReleaseSmokeTest(unittest.TestCase):
    def test_metrics_and_selection(self):
        np.testing.assert_allclose(
            tacs_from_loss_stack(np.array([[10.0, 4.0], [8.0, 3.0], [5.0, 2.0]])),
            [0.5, 0.5],
        )
        np.testing.assert_allclose(final_drop_from_loss_stack(np.array([[3.0, 1.5], [2.0, 1.0]])), [1.0, 0.5])
        np.testing.assert_allclose(average_ranks(np.array([2.0, 2.0, 1.0])), [2.5, 2.5, 1.0])
        self.assertEqual(binary_auc(np.array([0.1, 0.9, 0.8, 0.2]), np.array([False, True, True, False])), 1.0)
        self.assertEqual(topk_stats(np.array([0.9, 0.1, 0.8]), np.array([True, False, True]), 2)["purity"], 1.0)
        self.assertEqual(spearman_rank_correlation(np.array([3.0, 1.0, 2.0]), np.array([3.0, 1.0, 2.0])), 1.0)

    def test_matching_and_cli_help(self):
        scores = calculate_influence_score(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
        )
        self.assertTrue(torch.equal(scores, torch.tensor([[1.0, 1.0], [1.0, 0.0]])))

        selected = select_indices(torch.tensor([0.2, 0.9, 0.4]), [("a", 0), ("b", 0), ("a", 1)], top_k=2)
        self.assertEqual(selected[0][0:2], ("b", 0))
        self.assertEqual(selected[1][0:2], ("a", 1))

        for module in [
            "tacs.data_selection.matching",
            "tacs.data_selection.select_from_scores",
            "tacs.data_selection.write_selected_data",
        ]:
            result = subprocess.run(
                [sys.executable, "-m", module, "--help"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
