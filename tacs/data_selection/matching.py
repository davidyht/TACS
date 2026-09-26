import argparse
import os

import torch
import torch.nn.functional as F

from tacs.data_selection.validation_grouping import (
    GROUP_AGGREGATIONS,
    aggregate_validation_influence,
    load_validation_group_assignments,
)

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')
argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
                       help='The path to the gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="checkpoint weights")
argparser.add_argument('--target_task_names', type=str,
                       nargs='+', help="The name of the target tasks")
argparser.add_argument('--validation_gradient_path', type=str,
                       default="{} ckpt{}", help='The path to the validation gradient file')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')
argparser.add_argument(
    '--validation_group_file',
    type=str,
    default=None,
    help=(
        "Optional JSON manifest with one group assignment per validation example. "
        "Use {task} in the path for task-specific manifests. When omitted, retain "
        "LESS's metadata grouping (MMLU subject, BBH task, TyDiQA language)."
    ),
)
argparser.add_argument(
    '--validation_group_aggregation',
    choices=GROUP_AGGREGATIONS,
    default='max',
    help=(
        "How to combine group scores. rank_cvar uses within-group percentile "
        "ranks and averages the strongest fraction, reducing raw-max scale and "
        "single-group extreme-value effects."
    ),
)
argparser.add_argument('--validation_group_cvar_fraction', type=float, default=0.25)
argparser.add_argument('--validation_group_cvar_min_groups', type=int, default=2)


N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_info_path(path_or_dir: str) -> str:
    """Resolve a gradient/representation input to a concrete tensor file path."""
    if os.path.isdir(path_or_dir):
        for name in ("all_orig.pt", "all_unormalized.pt"):
            candidate = os.path.join(path_or_dir, name)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"Expected one of all_orig.pt/all_unormalized.pt under {path_or_dir}"
        )
    return path_or_dir


def _load_and_normalize_info(path_or_dir: str) -> torch.Tensor:
    """Load info tensor and row-normalize it for cosine-similarity scoring."""
    resolved = _resolve_info_path(path_or_dir)
    info = torch.load(resolved, map_location="cpu")
    if not torch.is_tensor(info):
        info = torch.tensor(info)
    info = info.float()
    # Enforce cosine-similarity semantics regardless of how artifacts were merged.
    info = F.normalize(info, p=2, dim=1, eps=1e-12)
    return info.to(device)


def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score.

    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
    """
    # N x N_VALID
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores


def main(argv=None):
    args = argparser.parse_args(argv)
    if not args.checkpoint_weights or not args.ckpts or not args.target_task_names or not args.train_file_names:
        argparser.error("--checkpoint_weights, --ckpts, --target_task_names, and --train_file_names are required")

    # renormalize the checkpoint weights
    if sum(args.checkpoint_weights) != 1:
        s = sum(args.checkpoint_weights)
        args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

    # calculate the influence score for each validation task
    for target_task_name in args.target_task_names:
        for train_file_name in args.train_file_names:
            influence_score = 0
            for i, ckpt in enumerate(args.ckpts):
                # validation_path = args.validation_gradient_path.format(
                # target_task_name, ckpt)
                validation_path = args.validation_gradient_path.format(
                    ckpt, target_task_name)
                validation_info = _load_and_normalize_info(validation_path)
                # gradient_path = args.gradient_path.format(train_file_name, ckpt)
                gradient_path = args.gradient_path.format(ckpt, train_file_name)
                training_info = _load_and_normalize_info(gradient_path)

                influence_score += args.checkpoint_weights[i] * \
                    calculate_influence_score(
                        training_info=training_info, validation_info=validation_info)
            if args.validation_group_file:
                group_path = args.validation_group_file.format(task=target_task_name)
                assignments = load_validation_group_assignments(
                    group_path,
                    expected_n=int(influence_score.shape[1]),
                    task=target_task_name,
                )
                influence_score, grouped_scores, group_labels = aggregate_validation_influence(
                    influence_score,
                    assignments,
                    method=args.validation_group_aggregation,
                    cvar_fraction=args.validation_group_cvar_fraction,
                    cvar_min_groups=args.validation_group_cvar_min_groups,
                )
                print(
                    f"Using learned validation groups from {group_path}: "
                    f"groups={len(group_labels)} sizes="
                    f"{[assignments.count(label) for label in group_labels]} "
                    f"aggregation={args.validation_group_aggregation}"
                )
                del grouped_scores
            else:
                influence_score = influence_score.reshape(
                    influence_score.shape[0], N_SUBTASKS[target_task_name], -1
                ).mean(-1).max(-1)[0]
            output_dir = os.path.join(args.output_path, target_task_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(
                args.output_path, target_task_name, f"{train_file_name}_influence_score.pt")
            torch.save(influence_score, output_file)
            print("Saved influence score to {}".format(output_file))


if __name__ == "__main__":
    main()
