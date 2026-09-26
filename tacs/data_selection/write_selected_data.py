import argparse
import os

import torch


def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--train_file_names', type=str,
                           nargs='+', help='The path to the score file')
    argparser.add_argument('--score_file_names', type=str,
                           nargs='+', default=None, help='Optional names to use for locating influence score files (defaults to train_file_names).')
    argparser.add_argument('--train_files', type=str, nargs='+',
                           help='The path of the training file that corresponds to the score file')
    argparser.add_argument('--target_task_names', type=str,
                           nargs='+', help='The name of the target task')
    argparser.add_argument('--output_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--max_samples', type=int,
                           default=None, help='The maximum number of samples')
    argparser.add_argument('--percentage', type=float, default=None,
                           help='The percentage of the data to be selected')

    args = argparser.parse_args()

    return args


def count_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        line_count = 0
        for line in file:
            line_count += 1
    return line_count


if __name__ == "__main__":
    args = parse_args()
    assert len(args.train_file_names) == len(args.train_files)
    if args.score_file_names is not None:
        assert len(args.score_file_names) == len(args.train_file_names), "score_file_names length must match train_file_names"
    assert args.percentage is not None or args.max_samples is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train_files = len(args.train_file_names)

    # Validate training files exist to avoid confusing downstream errors
    missing_train_files = [p for p in args.train_files if not os.path.exists(p)]
    if len(missing_train_files) > 0:
        raise FileNotFoundError(f"The following training files were not found: {missing_train_files}. Ensure paths are correct (e.g., use '*_data.jsonl' if that is the actual filename).")

    # Helper to resolve score file path with sensible fallback
    def resolve_score_path(base_output_path: str, name: str):
        primary = os.path.join(base_output_path, f"{name}_influence_score.pt")
        if os.path.exists(primary):
            return primary
        # Fallback: strip common suffix '_data'
        if name.endswith('_data'):
            alt_name = name[:-5]
            alt = os.path.join(base_output_path, f"{alt_name}_influence_score.pt")
            if os.path.exists(alt):
                return alt
        # If neither exists, return the primary path (load will error with clear message below)
        return primary

    for target_task in args.target_task_names:
        output_path = os.path.join(args.output_path, target_task)

        score_names = args.score_file_names if args.score_file_names is not None else args.train_file_names
        score_paths = [resolve_score_path(output_path, task_name) for task_name in score_names]
        num_samples = []
        for score_path in score_paths:
            if not os.path.exists(score_path):
                raise FileNotFoundError(f"Influence score file not found: '{score_path}'. If your saved file uses a base name without '_data', pass '--score_file_names' matching the saved influence score basenames (e.g., 'oasst1' instead of 'oasst1_data').")
            num_samples.append(len(torch.load(score_path, map_location=device)))
        cumsum_num_samples = torch.cumsum(torch.tensor(num_samples), dim=0)

        total_samples = sum(num_samples)
        if args.percentage is not None:
            args.max_samples = int(args.percentage * total_samples)
            data_amount_name = f"p{args.percentage}"
        else:
            data_amount_name = f"num{args.max_samples}"

        all_scores = []
        for score_path, train_file in zip(score_paths, args.train_files):
            score = torch.load(score_path, map_location=device)
            all_scores.append(score)
        all_scores = torch.cat(all_scores, dim=0)

        # sort the scores and output the corresponding data index
        file_specific_index = torch.cat(
            [torch.arange(line_num) for line_num in num_samples]).to(device)
        data_from = torch.cat([torch.ones(line_num, dtype=torch.long)
                              * i for i, line_num in enumerate(num_samples)]).to(device)
        sorted_scores, sorted_index = torch.sort(
            all_scores, dim=0, descending=True)
        sorted_score_file = os.path.join(output_path, f"sorted.csv")

        data_from = data_from[sorted_index]
        sorted_index = file_specific_index[sorted_index]


        if not os.path.exists(sorted_score_file):
            with open(sorted_score_file, 'w', encoding='utf-8') as file:
                file.write("file name, index, score\n")
                for score, index, name in zip(sorted_scores, sorted_index, data_from):
                    file.write(
                        f"{args.train_file_names[name.item()]}, {index.item()}, {round(score.item(), 6)}\n")

        topk_scores, topk_indices = torch.topk(
            all_scores.float(), args.max_samples, dim=0, largest=True)

        all_lines = []
        for i, train_file in enumerate(args.train_files):
            with open(train_file, 'r', encoding='utf-8', errors='ignore') as file:
                all_lines.append(file.readlines()[:num_samples[i]])

        final_index_list = sorted_index[:args.max_samples].tolist()
        final_data_from = data_from[:args.max_samples].tolist()
        with open(os.path.join(output_path, f"top_{data_amount_name}.jsonl"), 'w', encoding='utf-8', errors='ignore') as file:
            for index, data_from in zip(final_index_list, final_data_from):
                try:
                    file.write(all_lines[data_from][index])
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to write selected data: source_file_index={data_from}, line_index={index}. "
                        "Check that each train file has at least as many lines as its influence score length."
                    ) from exc
