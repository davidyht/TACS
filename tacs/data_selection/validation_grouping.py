"""Utilities for aggregating validation influence by learned or metadata groups."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch


GROUP_AGGREGATIONS = ("max", "mean", "rank_mean", "rank_cvar")


def _extract_assignments(payload: Any, task: str | None = None) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("validation-group manifest must be a list or JSON object")

    if "tasks" in payload:
        tasks = payload["tasks"]
        if not isinstance(tasks, dict) or task is None or task not in tasks:
            raise ValueError(f"validation-group manifest has no task entry for {task!r}")
        return _extract_assignments(tasks[task], task=task)
    if "assignments" in payload:
        assignments = payload["assignments"]
        if not isinstance(assignments, list):
            raise ValueError("validation-group manifest field 'assignments' must be a list")
        return assignments
    raise ValueError("validation-group manifest has no 'assignments' field")


def load_validation_group_assignments(
    path: str | Path,
    *,
    expected_n: int,
    task: str | None = None,
) -> list[Any]:
    """Load one group label per validation example from a JSON manifest."""
    manifest_path = Path(path).expanduser()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignments = _extract_assignments(payload, task=task)
    if len(assignments) != int(expected_n):
        raise ValueError(
            f"{manifest_path}: expected {expected_n} assignments, found {len(assignments)}"
        )
    if any(label is None or isinstance(label, (dict, list)) for label in assignments):
        raise ValueError(f"{manifest_path}: group labels must be JSON scalars")
    if not assignments:
        raise ValueError(f"{manifest_path}: empty assignments")
    return assignments


def percentile_rank_columns(grouped: torch.Tensor) -> torch.Tensor:
    """Return scale-invariant percentile ranks within every group column.

    Ties receive their average rank. A constant group maps every candidate to
    0.5, so that group cannot win merely through its raw numerical scale.
    """
    if grouped.ndim != 2:
        raise ValueError(f"grouped scores must be rank 2, got shape {tuple(grouped.shape)}")
    n_candidates = int(grouped.shape[0])
    if n_candidates < 1:
        raise ValueError("grouped scores must contain at least one candidate")

    columns = []
    for column in grouped.T:
        _, inverse, counts = torch.unique(
            column,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        ends = counts.cumsum(dim=0).to(dtype=grouped.dtype) - 1.0
        starts = ends - counts.to(dtype=grouped.dtype) + 1.0
        average_ranks = 0.5 * (starts + ends)
        if n_candidates == 1:
            percentiles = torch.full_like(average_ranks, 0.5)
        else:
            percentiles = average_ranks / float(n_candidates - 1)
        columns.append(percentiles[inverse])
    return torch.stack(columns, dim=1)


def aggregate_group_scores(
    grouped: torch.Tensor,
    *,
    method: str = "max",
    cvar_fraction: float = 0.25,
    cvar_min_groups: int = 2,
) -> torch.Tensor:
    """Aggregate candidate-by-group scores into one candidate score.

    ``rank_cvar`` converts each group to candidate percentiles, then averages
    a candidate's strongest fraction of groups. It is invariant to per-group
    scale and can require support from more than one group.
    """
    if grouped.ndim != 2 or grouped.shape[0] < 1 or grouped.shape[1] < 1:
        raise ValueError(f"grouped scores must be non-empty rank 2, got {tuple(grouped.shape)}")
    if method not in GROUP_AGGREGATIONS:
        raise ValueError(f"unknown group aggregation {method!r}; choose from {GROUP_AGGREGATIONS}")
    if method == "max":
        return grouped.max(dim=1).values
    if method == "mean":
        return grouped.mean(dim=1)

    ranked = percentile_rank_columns(grouped)
    if method == "rank_mean":
        return ranked.mean(dim=1)

    if not 0.0 < cvar_fraction <= 1.0:
        raise ValueError("cvar_fraction must be in (0, 1]")
    if cvar_min_groups < 1:
        raise ValueError("cvar_min_groups must be positive")
    n_groups = int(grouped.shape[1])
    top_groups = min(
        n_groups,
        max(int(cvar_min_groups), int(math.ceil(float(cvar_fraction) * n_groups))),
    )
    return ranked.topk(top_groups, dim=1).values.mean(dim=1)


def aggregate_validation_influence(
    influence: torch.Tensor,
    assignments: Sequence[Any],
    *,
    method: str = "max",
    cvar_fraction: float = 0.25,
    cvar_min_groups: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
    """Mean within validation groups, then aggregate the group scores.

    Args:
        influence: Candidate-by-validation-example influence matrix.
        assignments: One group label per validation-example column.

    Returns:
        ``(candidate_scores, candidate_by_group_scores, group_labels)``.
        Group labels follow first occurrence order, making the result stable
        for arbitrary string or integer labels.
    """
    if influence.ndim != 2:
        raise ValueError(f"influence must be rank 2, got shape {tuple(influence.shape)}")
    if influence.shape[1] != len(assignments):
        raise ValueError(
            f"influence has {influence.shape[1]} validation columns, "
            f"but {len(assignments)} assignments were provided"
        )

    group_labels: list[Any] = []
    group_to_indices: dict[Any, list[int]] = {}
    for idx, label in enumerate(assignments):
        try:
            indices = group_to_indices.get(label)
        except TypeError as exc:
            raise ValueError(f"unhashable validation-group label at index {idx}: {label!r}") from exc
        if indices is None:
            group_labels.append(label)
            group_to_indices[label] = [idx]
        else:
            indices.append(idx)

    grouped = torch.stack(
        [
            influence[:, group_to_indices[label]].mean(dim=1)
            for label in group_labels
        ],
        dim=1,
    )
    candidate_scores = aggregate_group_scores(
        grouped,
        method=method,
        cvar_fraction=cvar_fraction,
        cvar_min_groups=cvar_min_groups,
    )
    return candidate_scores, grouped, group_labels
