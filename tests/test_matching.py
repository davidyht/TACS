import torch

from tacs.data_selection.matching import calculate_influence_score


def test_calculate_influence_score_matrix_product():
    training = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    validation = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    scores = calculate_influence_score(training, validation)
    assert torch.equal(scores, torch.tensor([[1.0, 1.0], [1.0, 0.0]]))
