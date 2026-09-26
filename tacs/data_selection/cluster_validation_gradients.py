#!/usr/bin/env python3
"""Cluster validation-gradient rows with deterministic spherical k-means.

The intended input is LESS's row-normalized ``all_orig.pt`` artifact from one
or more warmup checkpoints. A row may represent an example or an already
aggregated subtask such as one MMLU subject. With multiple checkpoints, the
row-normalized vectors are concatenated and normalized once more, so cosine
similarity in the result is the average checkpoint-wise cosine similarity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gradient_tensor(path: str | Path) -> torch.Tensor:
    source = Path(path).expanduser()
    try:
        value: Any = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(source, map_location="cpu")
    if isinstance(value, dict):
        for key in ("grads", "gradients", "all_orig"):
            if key in value:
                value = value[key]
                break
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] < 2 or tensor.shape[1] < 1:
        raise ValueError(f"{source}: expected an N x D tensor, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{source}: gradient tensor contains non-finite values")
    norms = tensor.norm(dim=1)
    if torch.any(norms <= 0):
        count = int((norms <= 0).sum().item())
        raise ValueError(f"{source}: found {count} zero-norm gradient rows")
    return F.normalize(tensor, p=2, dim=1)


def residualize_by_label(features: torch.Tensor, labels: Sequence[str]) -> torch.Tensor:
    """Subtract each nuisance-label centroid, then row-normalize."""
    if features.shape[0] != len(labels):
        raise ValueError("nuisance-label count does not match feature rows")
    residual = features.clone()
    for label in dict.fromkeys(labels):
        indices = [idx for idx, value in enumerate(labels) if value == label]
        index = torch.tensor(indices, dtype=torch.long)
        residual[index] -= residual[index].mean(dim=0, keepdim=True)
    norms = residual.norm(dim=1)
    if torch.any(norms <= 1e-12):
        raise ValueError("nuisance residualization produced a zero-norm row")
    return F.normalize(residual, p=2, dim=1)


def build_path_features(
    gradient_tensors: Sequence[torch.Tensor],
    *,
    nuisance_labels: Sequence[str] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if not gradient_tensors:
        raise ValueError("at least one gradient tensor is required")
    n_rows = gradient_tensors[0].shape[0]
    if any(tensor.shape[0] != n_rows for tensor in gradient_tensors):
        raise ValueError("all checkpoint gradient tensors must have the same row count")

    prepared: list[torch.Tensor] = []
    for tensor in gradient_tensors:
        normalized = F.normalize(tensor.float(), p=2, dim=1)
        if nuisance_labels is not None:
            normalized = residualize_by_label(normalized, nuisance_labels)
        prepared.append(normalized)
    scale = math.sqrt(float(len(prepared)))
    path_features = F.normalize(torch.cat([tensor / scale for tensor in prepared], dim=1), p=2, dim=1)
    return path_features, prepared


def _kmeans_plus_plus(features: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    n_rows = features.shape[0]
    first = int(torch.randint(n_rows, (1,), generator=generator).item())
    indices = [first]
    best_similarity = features @ features[first]
    for _ in range(1, k):
        distances = torch.clamp(1.0 - best_similarity, min=0.0)
        distances[torch.tensor(indices, dtype=torch.long)] = 0.0
        total = float(distances.sum().item())
        if total <= 0:
            remaining = [idx for idx in range(n_rows) if idx not in set(indices)]
            next_idx = remaining[int(torch.randint(len(remaining), (1,), generator=generator).item())]
        else:
            next_idx = int(torch.multinomial(distances, 1, generator=generator).item())
        indices.append(next_idx)
        best_similarity = torch.maximum(best_similarity, features @ features[next_idx])
    return features[torch.tensor(indices, dtype=torch.long)].clone()


def _canonicalize_assignments(assignments: torch.Tensor) -> torch.Tensor:
    members = []
    for cluster_id in sorted(int(value) for value in assignments.unique().tolist()):
        cluster_members = torch.nonzero(assignments == cluster_id, as_tuple=False).flatten()
        members.append((int(cluster_members.min().item()), cluster_id))
    remap = {old: new for new, (_, old) in enumerate(sorted(members))}
    return torch.tensor([remap[int(value)] for value in assignments.tolist()], dtype=torch.long)


def spherical_kmeans(
    features: torch.Tensor,
    *,
    k: int,
    seed: int = 0,
    num_restarts: int = 20,
    max_iter: int = 200,
    min_cluster_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return assignments, normalized centers, and mean assigned cosine."""
    if features.ndim != 2:
        raise ValueError("features must be rank 2")
    features = F.normalize(features.float(), p=2, dim=1)
    n_rows = features.shape[0]
    if not 1 < k <= n_rows:
        raise ValueError(f"k must be in [2, {n_rows}], got {k}")
    if min_cluster_size < 1 or k * min_cluster_size > n_rows:
        raise ValueError("min_cluster_size is infeasible")
    if num_restarts < 1 or max_iter < 1:
        raise ValueError("num_restarts and max_iter must be positive")

    best: tuple[torch.Tensor, torch.Tensor, float] | None = None
    rejected_sizes: list[list[int]] = []
    for restart in range(num_restarts):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 1009 * restart)
        centers = _kmeans_plus_plus(features, k, generator)
        previous = None
        for _ in range(max_iter):
            similarities = features @ centers.T
            assignments = similarities.argmax(dim=1)
            if previous is not None and torch.equal(assignments, previous):
                break
            previous = assignments.clone()
            new_centers = []
            assigned_similarity = similarities.gather(1, assignments[:, None]).squeeze(1)
            for cluster_id in range(k):
                members = features[assignments == cluster_id]
                if members.shape[0] == 0:
                    replacement = int(assigned_similarity.argmin().item())
                    center = features[replacement]
                else:
                    center = members.mean(dim=0)
                new_centers.append(F.normalize(center, p=2, dim=0))
            centers = torch.stack(new_centers)

        similarities = features @ centers.T
        assignments = similarities.argmax(dim=1)
        sizes = torch.bincount(assignments, minlength=k).tolist()
        if min(sizes) < min_cluster_size:
            rejected_sizes.append([int(size) for size in sizes])
            continue
        objective = float(similarities.gather(1, assignments[:, None]).mean().item())
        if best is None or objective > best[2]:
            best = (assignments.clone(), centers.clone(), objective)

    if best is None:
        sample = rejected_sizes[:5]
        raise ValueError(
            f"no spherical-k-means restart met min_cluster_size={min_cluster_size}; "
            f"sample cluster sizes={sample}"
        )

    assignments = _canonicalize_assignments(best[0])
    centers = torch.stack(
        [
            F.normalize(features[assignments == cluster_id].mean(dim=0), p=2, dim=0)
            for cluster_id in range(k)
        ]
    )
    objective = float((features @ centers.T).gather(1, assignments[:, None]).mean().item())
    return assignments, centers, objective


def adjusted_rand_index(left: Sequence[int], right: Sequence[int]) -> float:
    """Adjusted Rand index without a scikit-learn dependency."""
    if len(left) != len(right) or not left:
        raise ValueError("ARI inputs must have the same non-zero length")
    left_values = list(dict.fromkeys(int(value) for value in left))
    right_values = list(dict.fromkeys(int(value) for value in right))
    table = torch.zeros((len(left_values), len(right_values)), dtype=torch.int64)
    left_map = {value: idx for idx, value in enumerate(left_values)}
    right_map = {value: idx for idx, value in enumerate(right_values)}
    for lval, rval in zip(left, right):
        table[left_map[int(lval)], right_map[int(rval)]] += 1

    def choose2(value: int) -> int:
        return value * (value - 1) // 2

    n_rows = len(left)
    sum_cells = sum(choose2(int(value)) for value in table.flatten().tolist())
    sum_left = sum(choose2(int(value)) for value in table.sum(dim=1).tolist())
    sum_right = sum(choose2(int(value)) for value in table.sum(dim=0).tolist())
    total = choose2(n_rows)
    if total == 0:
        return 1.0
    expected = (sum_left * sum_right) / total
    maximum = 0.5 * (sum_left + sum_right)
    denominator = maximum - expected
    if abs(denominator) < 1e-12:
        return 1.0 if sum_cells == maximum else 0.0
    return float((sum_cells - expected) / denominator)


def cosine_silhouette(features: torch.Tensor, assignments: torch.Tensor) -> float:
    distances = torch.clamp(1.0 - features @ features.T, min=0.0)
    values = []
    for idx in range(features.shape[0]):
        own = assignments == assignments[idx]
        own[idx] = False
        if int(own.sum().item()) == 0:
            values.append(0.0)
            continue
        a = float(distances[idx, own].mean().item())
        b = min(
            float(distances[idx, assignments == cluster_id].mean().item())
            for cluster_id in assignments.unique().tolist()
            if int(cluster_id) != int(assignments[idx])
        )
        denom = max(a, b)
        values.append(0.0 if denom <= 0 else (b - a) / denom)
    return float(sum(values) / len(values))


def load_nuisance_labels(path: str | Path, expected_n: int) -> list[str]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("labels")
    if not isinstance(payload, list) or len(payload) != expected_n:
        raise ValueError(f"nuisance-label file must contain exactly {expected_n} labels")
    return [str(value) for value in payload]


def load_mmlu_metadata(data_dir: str | Path, n_shot: int) -> list[dict[str, Any]]:
    root = Path(data_dir).expanduser()
    mmlu_root = root / "eval" / "mmlu" if (root / "eval" / "mmlu").is_dir() else root
    test_dir = mmlu_root / "test"
    dev_dir = mmlu_root / "dev"
    subjects = sorted(path.name[: -len("_test.csv")] for path in test_dir.glob("*_test.csv"))
    if not subjects:
        raise FileNotFoundError(f"no MMLU test subjects under {test_dir}")
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        dev_path = dev_dir / f"{subject}_dev.csv"
        with dev_path.open("r", encoding="utf-8", newline="") as handle:
            examples = list(csv.reader(handle))
        if len(examples) < n_shot:
            raise ValueError(f"{dev_path}: expected at least {n_shot} examples")
        for example_idx, example in enumerate(examples[:n_shot]):
            if not example:
                raise ValueError(f"{dev_path}: empty example row {example_idx}")
            rows.append(
                {
                    "row_index": len(rows),
                    "subject": subject,
                    "subject_example_index": example_idx,
                    "example_key": f"{subject}:{example_idx}",
                    "answer": str(example[-1]),
                }
            )
    return rows


def load_mmlu_subject_metadata(data_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(data_dir).expanduser()
    mmlu_root = root / "eval" / "mmlu" if (root / "eval" / "mmlu").is_dir() else root
    test_dir = mmlu_root / "test"
    subjects = sorted(path.name[: -len("_test.csv")] for path in test_dir.glob("*_test.csv"))
    if not subjects:
        raise FileNotFoundError(f"no MMLU test subjects under {test_dir}")
    return [
        {
            "row_index": idx,
            "subject": subject,
            "example_key": subject,
        }
        for idx, subject in enumerate(subjects)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num-restarts", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--nuisance-labels", type=Path)
    parser.add_argument("--mmlu-data-dir", type=Path)
    parser.add_argument(
        "--mmlu-row-unit",
        choices=["example", "subject"],
        default="example",
        help="Interpret each MMLU gradient row as an example or an aggregated subject.",
    )
    parser.add_argument("--mmlu-n-shot", type=int, default=5)
    parser.add_argument("--residualize-mmlu-answer", action="store_true")
    args = parser.parse_args()

    gradient_paths = [path.expanduser().resolve() for path in args.gradient_files]
    tensors = [load_gradient_tensor(path) for path in gradient_paths]
    n_rows = tensors[0].shape[0]

    metadata = None
    if args.mmlu_data_dir is not None:
        metadata = (
            load_mmlu_subject_metadata(args.mmlu_data_dir)
            if args.mmlu_row_unit == "subject"
            else load_mmlu_metadata(args.mmlu_data_dir, args.mmlu_n_shot)
        )
        if len(metadata) != n_rows:
            raise ValueError(
                f"MMLU metadata has {len(metadata)} rows but gradients have {n_rows}"
            )

    nuisance_labels = None
    nuisance_source = None
    if args.nuisance_labels is not None:
        nuisance_labels = load_nuisance_labels(args.nuisance_labels, n_rows)
        nuisance_source = str(args.nuisance_labels.expanduser().resolve())
    if args.residualize_mmlu_answer:
        if metadata is None:
            raise ValueError("--residualize-mmlu-answer requires --mmlu-data-dir")
        if args.mmlu_row_unit != "example":
            raise ValueError("--residualize-mmlu-answer requires --mmlu-row-unit example")
        if nuisance_labels is not None:
            raise ValueError("choose either --nuisance-labels or --residualize-mmlu-answer")
        nuisance_labels = [str(row["answer"]) for row in metadata]
        nuisance_source = "mmlu_answer"

    features, checkpoint_features = build_path_features(
        tensors,
        nuisance_labels=nuisance_labels,
    )
    assignments, _, objective = spherical_kmeans(
        features,
        k=args.num_clusters,
        seed=args.seed,
        num_restarts=args.num_restarts,
        max_iter=args.max_iter,
        min_cluster_size=args.min_cluster_size,
    )

    checkpoint_stability = []
    for checkpoint_idx, checkpoint_tensor in enumerate(checkpoint_features):
        checkpoint_assignments, _, checkpoint_objective = spherical_kmeans(
            checkpoint_tensor,
            k=args.num_clusters,
            seed=args.seed,
            num_restarts=args.num_restarts,
            max_iter=args.max_iter,
            # Only the path clustering defines downstream groups. A tiny
            # single-checkpoint cluster is itself useful instability evidence,
            # so do not reject it before ARI can be reported.
            min_cluster_size=1,
        )
        checkpoint_stability.append(
            {
                "checkpoint_index": checkpoint_idx,
                "path": str(gradient_paths[checkpoint_idx]),
                "objective_mean_cosine": checkpoint_objective,
                "ari_vs_path_clustering": adjusted_rand_index(
                    assignments.tolist(),
                    checkpoint_assignments.tolist(),
                ),
            }
        )

    clusters = []
    for cluster_id in range(args.num_clusters):
        members = torch.nonzero(assignments == cluster_id, as_tuple=False).flatten().tolist()
        entry: dict[str, Any] = {
            "cluster_id": cluster_id,
            "size": len(members),
            "member_indices": members,
        }
        if metadata is not None:
            entry["example_keys"] = [metadata[idx]["example_key"] for idx in members]
            subject_counts: dict[str, int] = {}
            answer_counts: dict[str, int] = {}
            for idx in members:
                subject = str(metadata[idx]["subject"])
                subject_counts[subject] = subject_counts.get(subject, 0) + 1
                if "answer" in metadata[idx]:
                    answer = str(metadata[idx]["answer"])
                    answer_counts[answer] = answer_counts.get(answer, 0) + 1
            entry["subject_counts"] = dict(sorted(subject_counts.items()))
            if answer_counts:
                entry["answer_counts"] = dict(sorted(answer_counts.items()))
        clusters.append(entry)

    manifest = {
        "version": 1,
        "method": "spherical_kmeans_validation_gradient_path",
        "num_examples": n_rows,
        "num_rows": n_rows,
        "row_unit": args.mmlu_row_unit if metadata is not None else "unspecified",
        "num_clusters": int(args.num_clusters),
        "seed": int(args.seed),
        "num_restarts": int(args.num_restarts),
        "max_iter": int(args.max_iter),
        "min_cluster_size": int(args.min_cluster_size),
        "gradient_files": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "shape": list(tensor.shape),
            }
            for path, tensor in zip(gradient_paths, tensors)
        ],
        "feature_construction": "concat_equal_weight_row_normalized_checkpoints_then_row_normalize",
        "nuisance_residualization": nuisance_source,
        "objective_mean_cosine": objective,
        "cosine_silhouette": cosine_silhouette(features, assignments),
        "assignments": [int(value) for value in assignments.tolist()],
        "cluster_sizes": [int(value) for value in torch.bincount(assignments).tolist()],
        "checkpoint_stability": checkpoint_stability,
        "clusters": clusters,
        "examples": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "output": str(args.output),
            "cluster_sizes": manifest["cluster_sizes"],
            "objective_mean_cosine": objective,
            "cosine_silhouette": manifest["cosine_silhouette"],
            "checkpoint_ari": [
                round(float(row["ari_vs_path_clustering"]), 4)
                for row in checkpoint_stability
            ],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
