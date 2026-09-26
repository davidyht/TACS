import numpy as np

from tacs.metrics import final_drop_from_loss_stack, tacs_from_loss_stack


def test_tacs_from_loss_stack_normalizes_endpoint_drop():
    losses = np.array([[10.0, 4.0], [8.0, 3.0], [5.0, 2.0]])
    np.testing.assert_allclose(tacs_from_loss_stack(losses), [0.5, 0.5])


def test_final_drop_from_loss_stack():
    losses = np.array([[3.0, 1.5], [2.0, 1.0]])
    np.testing.assert_allclose(final_drop_from_loss_stack(losses), [1.0, 0.5])


def test_tacs_requires_two_dimensional_stack():
    try:
        tacs_from_loss_stack(np.array([1.0, 2.0]))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
