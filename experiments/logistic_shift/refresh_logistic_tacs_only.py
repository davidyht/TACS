#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

import run_logistic_tov_style_comparison as logistic_exp


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fill_tacs_defaults(config: dict, *, tacs_score_window: str, tacs_hp_search: bool) -> dict:
    default_cfg = logistic_exp.build_config(
        quick=config.get("train_size") != 128 * 1024,
        val_size_override=config["val_size"],
        seed_count_override=len(config["seeds"]),
        tacs_score_window=tacs_score_window,
        tacs_hp_search=tacs_hp_search,
    )
    out = copy.deepcopy(config)
    for key in [
        "proxy_size",
        "tacs_score_window",
        "tacs_hp_search",
        "tacs_hp_pool_probe_size",
        "tacs_hp_steps_grid",
        "tacs_hp_lr_factor_grid",
    ]:
        out[key] = copy.deepcopy(default_cfg[key])
    return out


def refresh_tacs_only(source_json: Path, out_json: Path, *, tacs_score_window: str, disable_tacs_hp_search: bool) -> dict:
    start = time.time()
    original = load_json(source_json)
    config = fill_tacs_defaults(
        original["config"],
        tacs_score_window=tacs_score_window,
        tacs_hp_search=not disable_tacs_hp_search,
    )

    refreshed_seed_results = []
    for seed_payload in original["seed_results"]:
        seed = int(seed_payload["seed"])
        problem = logistic_exp.make_problem(config, seed)
        params0 = logistic_exp.init_params(config["dim"], config["init_scale"], np.random.default_rng(seed + 11_111))
        tacs_hp = logistic_exp.search_tacs_hparams(params0, problem, config, seed)
        scores = logistic_exp.score_tacs_from_base_val_trajectory(
            params0,
            problem,
            tacs_hp["lrs"],
            score_window=config["tacs_score_window"],
        )

        candidate_idx = problem["candidate_idx"]
        target_mask = problem["source_target"][candidate_idx]
        score_payload = {
            "auc_target_vs_other": float("nan") if target_mask.sum() == 0 else logistic_exp.binary_auc(scores, target_mask)
        }

        new_seed_payload = copy.deepcopy(seed_payload)
        new_seed_payload["tacs_hp_search"] = {
            "score_window": config["tacs_score_window"],
            "best": tacs_hp["best"],
            "grid": tacs_hp["grid"],
        }
        for strategy in logistic_exp.STRATEGIES:
            method_payload = {
                "score": score_payload,
                "k": {},
            }
            for k in config["budgets"]:
                selected_idx = logistic_exp.select_indices(
                    method="TACS",
                    strategy=strategy,
                    scores=scores,
                    problem=problem,
                    k=k,
                    seed=seed * 100_000 + k * 10 + logistic_exp.METHODS.index("TACS") * 1000 + logistic_exp.STRATEGIES.index(strategy),
                )
                method_payload["k"][str(k)] = logistic_exp.train_final_model(selected_idx, config, problem, seed + k)
            new_seed_payload["strategies"][strategy]["TACS"] = method_payload
        refreshed_seed_results.append(new_seed_payload)

    refreshed = copy.deepcopy(original)
    refreshed["config"] = config
    refreshed["seed_results"] = refreshed_seed_results
    refreshed["summary"] = logistic_exp.aggregate(
        refreshed_seed_results,
        refreshed["methods"],
        refreshed["strategies"],
        config["budgets"],
    )
    refreshed["tacs_refresh"] = {
        "source_json": str(source_json.resolve()),
        "score_window": config["tacs_score_window"],
        "tacs_hp_search": config["tacs_hp_search"],
        "elapsed_seconds": time.time() - start,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(refreshed, indent=2), encoding="utf-8")
    return refreshed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh only the TACS branch in an existing logistic comparison JSON.")
    parser.add_argument("--source-json", type=Path, required=True, help="Existing JSON with all baseline results.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the refreshed JSON.")
    parser.add_argument(
        "--tacs-score-window",
        choices=logistic_exp.TACS_SCORE_WINDOWS,
        default="ckpt1_to_last",
        help="Which checkpoints define the normalized TACS loss drop.",
    )
    parser.add_argument("--disable-tacs-hp-search", action="store_true", help="Skip TACS proxy-vs-pool HP search.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refreshed = refresh_tacs_only(
        args.source_json.resolve(),
        args.output.resolve(),
        tacs_score_window=args.tacs_score_window,
        disable_tacs_hp_search=args.disable_tacs_hp_search,
    )
    main_budget = str(refreshed["config"]["budgets"][-1])
    for strategy in refreshed["strategies"]:
        row = refreshed["summary"][strategy]["TACS"]["k"][main_budget]
        print(
            f"{strategy:>16} "
            f"err={row['classification_error_mean']:.3f}+-{row['classification_error_std']:.3f} "
            f"target_frac={row['selected_target_fraction_mean']:.3f}",
            flush=True,
        )
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
