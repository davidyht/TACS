#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared.metrics import binary_auc, mean_std  # noqa: E402
from _shared.tacs_core import tacs_from_loss_stack  # noqa: E402


OUT_PATH = Path(__file__).resolve().with_name("logistic_tov_style_comparison.json")
METHODS = [
    "TACS",
    "ToV_max_improv",
    "ToV_max_pos_improv",
    "ToV_max_abs_cng",
    "LESS_multi",
    "MaxUncert",
    "Random",
    "Oracle_source",
]
STRATEGIES = ["random_from_top", "score_plus_random", "score_only"]
STRATEGY_LABELS = {
    "random_from_top": "Rand-frm-top",
    "score_plus_random": "Score+Random",
    "score_only": "Score-only",
}
TACS_SCORE_WINDOWS = ["init_to_last", "ckpt1_to_last"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def clone_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"w": params["w"].copy(), "b": np.array(params["b"], dtype=np.float64)}


def init_params(dim: int, init_scale: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "w": rng.normal(0.0, init_scale, size=(dim,)),
        "b": np.array(0.0, dtype=np.float64),
    }


def logits(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return x @ params["w"] + params["b"]


def per_sample_losses(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    logit = logits(params, x)
    return np.logaddexp(0.0, logit) - y * logit


def mean_loss(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> float:
    return float(per_sample_losses(params, x, y).mean())


def mean_grad(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    err = sigmoid(logits(params, x)) - y
    n = max(len(x), 1)
    return {
        "w": (x * err[:, None]).sum(axis=0) / n,
        "b": np.array(err.mean(), dtype=np.float64),
    }


def gd_step(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, lr: float) -> dict[str, np.ndarray]:
    g = mean_grad(params, x, y)
    return {
        "w": params["w"] - lr * g["w"],
        "b": np.array(params["b"] - lr * g["b"], dtype=np.float64),
    }


def accuracy(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> float:
    pred = sigmoid(logits(params, x)) >= 0.5
    return float((pred == y).mean())


def classification_error(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> float:
    return 1.0 - accuracy(params, x, y)


def grad_dot_per_sample(
    params: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    g_ref: dict[str, np.ndarray],
) -> np.ndarray:
    err = sigmoid(logits(params, x)) - y
    return err * (x @ g_ref["w"] + g_ref["b"])


def uncertainty_score(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    p = sigmoid(logits(params, x))
    return p * (1.0 - p)


def make_linear_decay_schedule(epochs: int, lr0: float) -> list[float]:
    if epochs <= 0:
        return []
    return [lr0 * (1.0 - step / epochs) for step in range(epochs)]


def unit_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return v / norm


def make_target_and_distractor(config: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    theta_star = unit_vector(rng.normal(size=(config["dim"],)))
    raw = rng.normal(size=(config["dim"],))
    raw -= raw.dot(theta_star) * theta_star
    orth = unit_vector(raw)
    gamma = config["angle_rad"]
    theta_prime = unit_vector(math.cos(gamma) * theta_star + math.sin(gamma) * orth)
    return theta_star, theta_prime


def sample_logistic_distribution(
    rng: np.random.Generator,
    n: int,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(0.0, 1.0, size=(n, len(theta)))
    p = sigmoid(x @ theta)
    y = rng.binomial(1, p, size=n).astype(np.float64)
    return x, y


def build_budgets(start: int, end: int, mult: float) -> list[int]:
    budgets = []
    current = float(start)
    while current < end * 0.999999:
        budgets.append(int(round(current)))
        current *= mult
    budgets.append(int(end))
    out = sorted(set(budgets))
    return [k for k in out if k <= end]


def make_problem(config: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    theta_star, theta_prime = make_target_and_distractor(config, rng)

    n_target = config["train_size"] // 2
    n_distractor = config["train_size"] - n_target
    x_t, y_t = sample_logistic_distribution(rng, n_target, theta_star)
    x_d, y_d = sample_logistic_distribution(rng, n_distractor, theta_prime)
    x_train = np.concatenate([x_t, x_d], axis=0)
    y_train = np.concatenate([y_t, y_d], axis=0)
    source_target = np.concatenate(
        [
            np.ones(n_target, dtype=bool),
            np.zeros(n_distractor, dtype=bool),
        ],
        axis=0,
    )
    perm = rng.permutation(config["train_size"])
    x_train = x_train[perm]
    y_train = y_train[perm]
    source_target = source_target[perm]

    x_val, y_val = sample_logistic_distribution(rng, config["val_size"], theta_star)
    x_test, y_test = sample_logistic_distribution(rng, config["test_size"], theta_star)
    proxy_rng = np.random.default_rng(seed + 707_707)
    x_proxy, y_proxy = sample_logistic_distribution(proxy_rng, config["proxy_size"], theta_star)

    base_idx = rng.permutation(config["train_size"])[: config["base_size"]]
    in_base = np.zeros(config["train_size"], dtype=bool)
    in_base[base_idx] = True
    candidate_idx = np.flatnonzero(~in_base)

    return {
        "theta_star": theta_star,
        "theta_prime": theta_prime,
        "x_train": x_train,
        "y_train": y_train,
        "source_target": source_target,
        "x_val": x_val,
        "y_val": y_val,
        "x_proxy": x_proxy,
        "y_proxy": y_proxy,
        "x_test": x_test,
        "y_test": y_test,
        "base_idx": base_idx,
        "candidate_idx": candidate_idx,
    }


def train_trajectory(
    params0: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    lrs: list[float],
) -> list[dict[str, np.ndarray]]:
    params = clone_params(params0)
    checkpoints = []
    for lr in lrs:
        params = gd_step(params, x, y, lr)
        checkpoints.append(clone_params(params))
    return checkpoints


def score_tov_family(
    checkpoints: list[dict[str, np.ndarray]],
    problem: dict,
    val_lrs: list[float],
) -> dict[str, np.ndarray]:
    cand = problem["candidate_idx"]
    x_cand = problem["x_train"][cand]
    y_cand = problem["y_train"][cand]
    score_sum = {
        "ToV_max_improv": np.zeros(len(cand), dtype=np.float64),
        "ToV_max_pos_improv": np.zeros(len(cand), dtype=np.float64),
        "ToV_max_abs_cng": np.zeros(len(cand), dtype=np.float64),
    }
    for checkpoint, lr in zip(checkpoints, val_lrs):
        before = per_sample_losses(checkpoint, x_cand, y_cand)
        after = gd_step(checkpoint, problem["x_val"], problem["y_val"], lr)
        gpu = before - per_sample_losses(after, x_cand, y_cand)
        score_sum["ToV_max_improv"] += gpu
        score_sum["ToV_max_pos_improv"] += np.maximum(gpu, 0.0)
        score_sum["ToV_max_abs_cng"] += np.abs(gpu)
    for key in score_sum:
        score_sum[key] /= max(len(checkpoints), 1)
    return score_sum


def score_less_multi(
    checkpoints: list[dict[str, np.ndarray]],
    problem: dict,
) -> np.ndarray:
    cand = problem["candidate_idx"]
    x_cand = problem["x_train"][cand]
    y_cand = problem["y_train"][cand]
    score = np.zeros(len(cand), dtype=np.float64)
    for checkpoint in checkpoints:
        g_val = mean_grad(checkpoint, problem["x_val"], problem["y_val"])
        score += grad_dot_per_sample(checkpoint, x_cand, y_cand, g_val)
    return score / max(len(checkpoints), 1)


def score_tacs_on_eval_set(
    params0: dict[str, np.ndarray],
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    tacs_val_lrs: list[float],
    score_window: str,
) -> np.ndarray:
    params = clone_params(params0)
    losses = []
    if score_window == "init_to_last":
        losses.append(per_sample_losses(params, x_eval, y_eval))
    for lr in tacs_val_lrs:
        params = gd_step(params, x_val, y_val, lr)
        losses.append(per_sample_losses(params, x_eval, y_eval))
    if score_window not in TACS_SCORE_WINDOWS:
        raise ValueError(f"unknown TACS score window: {score_window}")
    if len(losses) < 2:
        raise ValueError("TACS requires at least two scored checkpoints")
    return tacs_from_loss_stack(np.stack(losses, axis=0), eps=1e-12)


def score_tacs_from_base_val_trajectory(
    params0: dict[str, np.ndarray],
    problem: dict,
    tacs_val_lrs: list[float],
    score_window: str,
) -> np.ndarray:
    cand = problem["candidate_idx"]
    return score_tacs_on_eval_set(
        params0,
        problem["x_val"],
        problem["y_val"],
        problem["x_train"][cand],
        problem["y_train"][cand],
        tacs_val_lrs,
        score_window=score_window,
    )


def build_tacs_hp_candidates(config: dict) -> list[dict[str, object]]:
    candidates = []
    seen = set()
    for steps in config["tacs_hp_steps_grid"]:
        for factor in config["tacs_hp_lr_factor_grid"]:
            key = (int(steps), float(factor))
            if key in seen:
                continue
            seen.add(key)
            lr0 = config["lr0"] * config["val_lr_scale"] * float(factor)
            candidates.append(
                {
                    "steps": int(steps),
                    "lr_factor": float(factor),
                    "lr0": float(lr0),
                    "lrs": make_linear_decay_schedule(int(steps), float(lr0)),
                }
            )
    return candidates


def search_tacs_hparams(
    params0: dict[str, np.ndarray],
    problem: dict,
    config: dict,
    seed: int,
) -> dict[str, object]:
    if not config["tacs_hp_search"]:
        default_lrs = make_linear_decay_schedule(config["tacs_val_epochs"], config["lr0"] * config["val_lr_scale"])
        return {
            "best": {
                "steps": int(config["tacs_val_epochs"]),
                "lr_factor": 1.0,
                "lr0": float(config["lr0"] * config["val_lr_scale"]),
                "proxy_vs_pool_auc": float("nan"),
                "proxy_score_mean": float("nan"),
                "pool_score_mean": float("nan"),
            },
            "grid": [],
            "lrs": default_lrs,
        }

    rng = np.random.default_rng(seed + 55_551)
    candidate_idx = problem["candidate_idx"]
    probe_size = min(config["tacs_hp_pool_probe_size"], len(candidate_idx))
    probe_idx = rng.permutation(candidate_idx)[:probe_size]
    x_probe = np.concatenate([problem["x_proxy"], problem["x_train"][probe_idx]], axis=0)
    y_probe = np.concatenate([problem["y_proxy"], problem["y_train"][probe_idx]], axis=0)
    proxy_mask = np.concatenate(
        [
            np.ones(len(problem["x_proxy"]), dtype=bool),
            np.zeros(len(probe_idx), dtype=bool),
        ],
        axis=0,
    )

    best_row = None
    best_lrs = None
    grid_rows = []
    for candidate in build_tacs_hp_candidates(config):
        scores = score_tacs_on_eval_set(
            params0,
            problem["x_val"],
            problem["y_val"],
            x_probe,
            y_probe,
            candidate["lrs"],
            score_window=config["tacs_score_window"],
        )
        row = {
            "steps": int(candidate["steps"]),
            "lr_factor": float(candidate["lr_factor"]),
            "lr0": float(candidate["lr0"]),
            "proxy_vs_pool_auc": float(binary_auc(scores, proxy_mask)),
            "proxy_score_mean": float(scores[: len(problem["x_proxy"])].mean()),
            "pool_score_mean": float(scores[len(problem["x_proxy"]) :].mean()),
        }
        grid_rows.append(row)
        better = best_row is None or row["proxy_vs_pool_auc"] > best_row["proxy_vs_pool_auc"] + 1e-12
        tie = best_row is not None and abs(row["proxy_vs_pool_auc"] - best_row["proxy_vs_pool_auc"]) <= 1e-12
        if better or (tie and (row["steps"], abs(row["lr_factor"] - 1.0)) < (best_row["steps"], abs(best_row["lr_factor"] - 1.0))):
            best_row = row
            best_lrs = candidate["lrs"]

    return {
        "best": best_row,
        "grid": grid_rows,
        "lrs": best_lrs,
    }


def score_max_uncert(checkpoints: list[dict[str, np.ndarray]], problem: dict) -> np.ndarray:
    cand = problem["candidate_idx"]
    x_cand = problem["x_train"][cand]
    score = np.zeros(len(cand), dtype=np.float64)
    for checkpoint in checkpoints:
        score += uncertainty_score(checkpoint, x_cand)
    return score / max(len(checkpoints), 1)


def source_oracle_scores(problem: dict, seed: int) -> np.ndarray:
    cand = problem["candidate_idx"]
    mask = problem["source_target"][cand].astype(np.float64)
    rng = np.random.default_rng(seed + 77_771)
    return mask + 1e-6 * rng.random(len(cand))


def random_scores(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 91_991)
    return rng.random(n)


def select_indices(
    method: str,
    strategy: str,
    scores: np.ndarray,
    problem: dict,
    k: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cand = problem["candidate_idx"]
    base = problem["base_idx"]
    train_size = len(problem["x_train"])

    if method == "Random":
        return rng.permutation(train_size)[:k]

    order = np.argsort(scores)[::-1]
    if strategy == "score_only":
        return cand[order[:k]]

    if strategy == "score_plus_random":
        top_count = min(k // 2, len(cand))
        rand_count = k - top_count
        top_idx = cand[order[:top_count]]
        if rand_count <= len(base):
            rand_idx = rng.permutation(base)[:rand_count]
        else:
            rand_base = rng.permutation(base)
            extra = rng.permutation(cand[order[top_count:]])[: rand_count - len(base)]
            rand_idx = np.concatenate([rand_base, extra], axis=0)
        return np.concatenate([top_idx, rand_idx], axis=0)

    if strategy == "random_from_top":
        top_pool_size = max(int(math.ceil(0.5 * len(cand))), k)
        top_pool = cand[order[:top_pool_size]]
        return rng.permutation(top_pool)[:k]

    raise ValueError(f"unknown strategy: {strategy}")


def train_final_model(
    selected_idx: np.ndarray,
    config: dict,
    problem: dict,
    seed: int,
) -> dict[str, float]:
    init_rng = np.random.default_rng(seed + 123_321)
    params0 = init_params(config["dim"], config["init_scale"], init_rng)
    lrs = make_linear_decay_schedule(config["train_epochs"], config["lr0"])
    trajectory = train_trajectory(
        params0,
        problem["x_train"][selected_idx],
        problem["y_train"][selected_idx],
        lrs,
    )
    final_params = trajectory[-1]
    return {
        "classification_error": classification_error(final_params, problem["x_test"], problem["y_test"]),
        "test_log_loss": mean_loss(final_params, problem["x_test"], problem["y_test"]),
        "selected_target_fraction": float(problem["source_target"][selected_idx].mean()),
    }


def aggregate(seed_results: list[dict], methods: list[str], strategies: list[str], budgets: list[int]) -> dict:
    summary: dict[str, dict] = {}
    for strategy in strategies:
        summary[strategy] = {}
        for method in methods:
            summary[strategy][method] = {
                "score": {},
                "k": {},
            }
            for metric in ["auc_target_vs_other"]:
                stats = mean_std(
                    row["strategies"][strategy][method]["score"][metric]
                    for row in seed_results
                )
                summary[strategy][method]["score"][f"{metric}_mean"] = stats["mean"]
                summary[strategy][method]["score"][f"{metric}_std"] = stats["std"]
            for k in budgets:
                row_key = str(k)
                summary[strategy][method]["k"][row_key] = {}
                for metric in ["classification_error", "test_log_loss", "selected_target_fraction"]:
                    stats = mean_std(
                        row["strategies"][strategy][method]["k"][row_key][metric]
                        for row in seed_results
                    )
                    summary[strategy][method]["k"][row_key][f"{metric}_mean"] = stats["mean"]
                    summary[strategy][method]["k"][row_key][f"{metric}_std"] = stats["std"]
    return summary


def build_config(
    quick: bool,
    val_size_override: int | None = None,
    seed_count_override: int | None = None,
    tacs_score_window: str = "ckpt1_to_last",
    tacs_hp_search: bool = True,
) -> dict:
    config = {
        "dim": 10,
        "angle_rad": math.pi / 2.0,
        "train_size": 128 * 1024,
        "base_size": 4 * 1024,
        "val_size": 1024,
        "proxy_size": 512,
        "test_size": 10_000,
        "train_epochs": 4,
        "lr0": 0.5,
        "val_lr_scale": 0.1,
        "tacs_val_epochs": 4,
        "init_scale": 0.0,
        "budgets": build_budgets(128, 8192, math.sqrt(2.0)),
        "seeds": list(range(10)),
        "tacs_trajectory_source": "base_model_on_val",
        "tacs_score_window": tacs_score_window,
        "tacs_hp_search": bool(tacs_hp_search),
        "tacs_hp_pool_probe_size": 512,
        "tacs_hp_steps_grid": [2, 4, 8],
        "tacs_hp_lr_factor_grid": [0.25, 0.5, 1.0, 2.0],
    }
    if quick:
        config.update(
            {
                "train_size": 32 * 1024,
                "base_size": 2 * 1024,
                "proxy_size": 256,
                "test_size": 3000,
                "budgets": build_budgets(128, 2048, math.sqrt(2.0)),
                "seeds": [0, 1, 2],
                "tacs_hp_pool_probe_size": 256,
            }
        )
    if val_size_override is not None:
        config["val_size"] = int(val_size_override)
    if seed_count_override is not None:
        config["seeds"] = list(range(int(seed_count_override)))
    return config


def run(config: dict, out_path: Path) -> dict:
    start = time.time()
    seed_results = []
    base_lrs = make_linear_decay_schedule(config["train_epochs"], config["lr0"])
    tov_val_lrs = [config["val_lr_scale"] * lr for lr in base_lrs]
    for seed in config["seeds"]:
        print(f"\n=== seed {seed} ===", flush=True)
        problem = make_problem(config, seed)
        params0 = init_params(config["dim"], config["init_scale"], np.random.default_rng(seed + 11_111))
        checkpoints = train_trajectory(
            params0,
            problem["x_train"][problem["base_idx"]],
            problem["y_train"][problem["base_idx"]],
            base_lrs,
        )
        cand = problem["candidate_idx"]

        scores = {}
        scores.update(score_tov_family(checkpoints, problem, tov_val_lrs))
        scores["LESS_multi"] = score_less_multi(checkpoints, problem)
        tacs_hp = search_tacs_hparams(params0, problem, config, seed)
        scores["TACS"] = score_tacs_from_base_val_trajectory(
            params0,
            problem,
            tacs_hp["lrs"],
            score_window=config["tacs_score_window"],
        )
        scores["MaxUncert"] = score_max_uncert(checkpoints, problem)
        scores["Random"] = random_scores(len(cand), seed)
        scores["Oracle_source"] = source_oracle_scores(problem, seed)

        score_payload = {}
        target_mask = problem["source_target"][cand]
        for method in METHODS:
            score_payload[method] = {
                "auc_target_vs_other": float("nan") if target_mask.sum() == 0 else binary_auc(scores[method], target_mask)
            }

        payload = {
            "seed": seed,
            "candidate_count": int(len(cand)),
            "candidate_target_fraction": float(target_mask.mean()),
            "base_target_fraction": float(problem["source_target"][problem["base_idx"]].mean()),
            "tacs_hp_search": {
                "score_window": config["tacs_score_window"],
                "best": tacs_hp["best"],
                "grid": tacs_hp["grid"],
            },
            "strategies": {},
        }

        for strategy in STRATEGIES:
            payload["strategies"][strategy] = {}
            for method in METHODS:
                method_payload = {
                    "score": score_payload[method],
                    "k": {},
                }
                for k in config["budgets"]:
                    selected_idx = select_indices(
                        method=method,
                        strategy=strategy,
                        scores=scores[method],
                        problem=problem,
                        k=k,
                        seed=seed * 100_000 + k * 10 + METHODS.index(method) * 1000 + STRATEGIES.index(strategy),
                    )
                    method_payload["k"][str(k)] = train_final_model(selected_idx, config, problem, seed + k)
                payload["strategies"][strategy][method] = method_payload

            main_k = str(config["budgets"][-1])
            tacs_err = payload["strategies"][strategy]["TACS"]["k"][main_k]["classification_error"]
            tov_err = payload["strategies"][strategy]["ToV_max_improv"]["k"][main_k]["classification_error"]
            less_err = payload["strategies"][strategy]["LESS_multi"]["k"][main_k]["classification_error"]
            print(
                f"{STRATEGY_LABELS[strategy]:>14} "
                f"TACS={tacs_err:.3f} "
                f"ToV={tov_err:.3f} "
                f"LESS={less_err:.3f}",
                flush=True,
            )
        best_hp = tacs_hp["best"]
        print(
            f"{'TACS-HP':>14} steps={best_hp['steps']} "
            f"lr_factor={best_hp['lr_factor']:.2f} "
            f"proxy_auc={best_hp['proxy_vs_pool_auc']:.3f}",
            flush=True,
        )

        seed_results.append(payload)

    result = {
        "experiment": "logistic_tov_style_comparison",
        "methods": METHODS,
        "strategies": STRATEGIES,
        "config": config,
        "elapsed_seconds": time.time() - start,
        "seed_results": seed_results,
        "summary": aggregate(seed_results, METHODS, STRATEGIES, config["budgets"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    main_k = str(config["budgets"][-1])
    for strategy in STRATEGIES:
        print(f"\n[{STRATEGY_LABELS[strategy]}]", flush=True)
        for method in ["TACS", "ToV_max_improv", "LESS_multi", "MaxUncert", "Random", "Oracle_source"]:
            row = result["summary"][strategy][method]["k"][main_k]
            print(
                f"{method:>18} "
                f"err={row['classification_error_mean']:.3f}+-{row['classification_error_std']:.3f} "
                f"target_frac={row['selected_target_fraction_mean']:.3f}",
                flush=True,
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ToV-style logistic regression comparison with multi-checkpoint baselines.")
    parser.add_argument("--quick", action="store_true", help="Run a smaller smoke test.")
    parser.add_argument("--val-size", type=int, default=None, help="Override the validation-set size.")
    parser.add_argument("--seed-count", type=int, default=None, help="Override the number of seeds to run.")
    parser.add_argument(
        "--tacs-score-window",
        choices=TACS_SCORE_WINDOWS,
        default="ckpt1_to_last",
        help="Which checkpoints define the normalized TACS loss drop.",
    )
    parser.add_argument("--disable-tacs-hp-search", action="store_true", help="Skip TACS proxy-vs-pool hyperparameter search.")
    parser.add_argument("--output", type=Path, default=OUT_PATH, help="Where to write the JSON results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        build_config(
            args.quick,
            val_size_override=args.val_size,
            seed_count_override=args.seed_count,
            tacs_score_window=args.tacs_score_window,
            tacs_hp_search=not args.disable_tacs_hp_search,
        ),
        args.output,
    )


if __name__ == "__main__":
    main()
