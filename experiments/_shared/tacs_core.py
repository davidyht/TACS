from __future__ import annotations

import numpy as np


def tacs_from_loss_stack(loss_stack: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    loss_stack = np.asarray(loss_stack, dtype=np.float64)
    if loss_stack.ndim != 2:
        raise ValueError(f"expected a 2D loss stack, got shape {loss_stack.shape}")
    if loss_stack.shape[0] < 2:
        raise ValueError("TACS requires at least two scored checkpoints")
    first = loss_stack[0]
    last = loss_stack[-1]
    return (first - last) / np.maximum(first, eps)


def final_drop_from_loss_stack(loss_stack: np.ndarray) -> np.ndarray:
    loss_stack = np.asarray(loss_stack, dtype=np.float64)
    if loss_stack.ndim != 2 or loss_stack.shape[0] < 2:
        raise ValueError(f"expected at least two checkpoints, got shape {loss_stack.shape}")
    return loss_stack[0] - loss_stack[-1]
