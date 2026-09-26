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

from _shared.metrics import binary_auc, mean_std, spearman_rank_correlation, topk_stats  # noqa: E402
from _shared.tacs_core import tacs_from_loss_stack  # noqa: E402


OUT_PATH = Path(__file__).resolve().with_name("logistic_shift_tacs_results.json")
METHODS = ["TACS", "ToV_1step", "LESS_tracein", "GradNorm", "Random"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def clone_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"w": params["w"].copy(), "b": np.array(params["b"], dtype=np.float64)}


def init_params(rng: np.random.Generator, dim: int, scale: float) -> dict[str, np.ndarray]:
    return {
        "w": rng.normal(0.0, scale, size=(dim,)),
        "b": np.array(0.0, dtype=np.float64),
    }


def logits(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return x @ params["w"] + params["b"]


def binary_losses_from_logits(logit: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, logit) - y * logit


def per_sample_losses(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return binary_losses_from_logits(logits(params, x), y)


def mean_loss(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> float:
    return float(per_sample_losses(params, x, y).mean())


def grad_mean(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    err = sigmoid(logits(params, x)) - y
    n = max(len(x), 1)
    return {
        "w": (x * err[:, None]).sum(axis=0) / n,
        "b": np.array(err.mean(), dtype=np.float64),
    }


def gd_step(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, lr: float) -> dict[str, np.ndarray]:
    g = grad_mean(params, x, y)
    return {
        "w": params["w"] - lr * g["w"],
        "b": np.array(params["b"] - lr * g["b"], dtype=np.float64),
    }


def grad_dot_per_sample(
    params: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    g_ref: dict[str, np.ndarray],
) -> np.ndarray:
    err = sigmoid(logits(params, x)) - y
    return err * (x @ g_ref["w"] + g_ref["b"])


def grad_norm_per_sample(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    err = sigmoid(logits(params, x)) - y
    return np.sqrt((err * err) * ((x * x).sum(axis=1) + 1.0))


def accuracy(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> float:
    pred = sigmoid(logits(params, x)) >= 0.5
    return float((pred == y).mean())


def make_schedule(steps: int, lr0: float) -> list[float]:
    if steps <= 0:
        return []
    return [lr0 * (1.0 - step / steps) for step in range(steps)]


def train_steps(
    params: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    lrs: list[float],
) -> dict[str, np.ndarray]:
    out = clone_params(params)
    for lr in lrs:
        out = gd_step(out, x, y, lr)
    return out


def unit_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return v / norm


def balanced_labels(rng: np.random.Generator, n: int) -> np.ndarray:
    y = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    rng.shuffle(y)
    return y.astype(np.float64)


def make_env_geometry(config: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_envs = 1 + config["n_distractors"]
    dim = config["dim"]
    directions = np.zeros((n_envs, dim), dtype=np.float64)
    offsets = np.zeros((n_envs, dim), dtype=np.float64)
    directions[0] = unit_vector(rng.normal(size=(dim,)))
    offsets[0] = unit_vector(rng.normal(size=(dim,)))
    angle = math.radians(config["distractor_angle_deg"])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    for env in range(1, n_envs):
        orth = rng.normal(size=(dim,))
        orth -= orth.dot(directions[0]) * directions[0]
        orth = unit_vector(orth)
        directions[env] = unit_vector(cos_a * directions[0] + sin_a * orth)
        offsets[env] = unit_vector(rng.normal(size=(dim,)))
    return directions, offsets


def sample_points(
    env_ids: np.ndarray,
    y: np.ndarray,
    directions: np.ndarray,
    offsets: np.ndarray,
    config: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    sign = 2.0 * y - 1.0
    signal = config["class_sep"] * sign[:, None] * directions[env_ids]
    env_shift = config["env_offset_scale"] * offsets[env_ids]
    noise = rng.normal(0.0, config["noise_std"], size=(len(env_ids), config["dim"]))
    return signal + env_shift + noise


def sample_pool_envs(config: dict, rng: np.random.Generator) -> np.ndarray:
    n_envs = 1 + config["n_distractors"]
    probs = np.full(n_envs, (1.0 - config["target_env_rate"]) / config["n_distractors"], dtype=np.float64)
    probs[0] = config["target_env_rate"]
    return rng.choice(np.arange(n_envs), size=config["train_size"], p=probs)


def make_problem(config: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    directions, offsets = make_env_geometry(config, rng)
    env_train = sample_pool_envs(config, rng)
    y_train = balanced_labels(rng, config["train_size"])
    x_train = sample_points(env_train, y_train, directions, offsets, config, rng)

    env_val = np.zeros(config["val_size"], dtype=np.int64)
    y_val = balanced_labels(rng, config["val_size"])
    x_val = sample_points(env_val, y_val, directions, offsets, config, rng)

    env_test = np.zeros(config["test_size"], dtype=np.int64)
    y_test = balanced_labels(rng, config["test_size"])
    x_test = sample_points(env_test, y_test, directions, offsets, config, rng)

    perm = rng.permutation(config["train_size"])
    base_idx = perm[: config["base_size"]]
    in_base = np.zeros(config["train_size"], dtype=bool)
    in_base[base_idx] = True
    candidate_idx = np.flatnonzero(~in_base)

    init_rng = np.random.default_rng(seed + 100_003)
    retrain_rng = np.random.default_rng(seed + 200_003)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "env_train": env_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "base_idx": base_idx,
        "candidate_idx": candidate_idx,
        "params0": init_params(init_rng, config["dim"], config["init_scale"]),
        "retrain_params0": init_params(retrain_rng, config["dim"], config["init_scale"]),
    }


def score_tacs(problem: dict, base_params: dict[str, np.ndarray], config: dict, val_lrs: list[float]) -> np.ndarray:
    params = clone_params(base_params)
    cand = problem["candidate_idx"]
    x_cand = problem["x_train"][cand]
    y_cand = problem["y_train"][cand]
    losses = [per_sample_losses(params, x_cand, y_cand)]
    for lr in val_lrs:
        params = gd_step(params, problem["x_val"], problem["y_val"], lr * config["val_lr_scale"])
        losses.append(per_sample_losses(params, x_cand, y_cand))
    return tacs_from_loss_stack(np.stack(losses, axis=0), eps=config["score_eps"])


def score_tov_1step(problem: dict, base_params: dict[str, np.ndarray], config: dict, val_lrs: list[float]) -> np.ndarray:
    cand = problem["candidate_idx"]
    x_cand = problem["x_train"][cand]
    y_cand = problem["y_train"][cand]
    before = per_sample_losses(base_params, x_cand, y_cand)
    after_params = gd_step(base_params, problem["x_val"], problem["y_val"], val_lrs[0] * config["val_lr_scale"])
    after = per_sample_losses(after_params, x_cand, y_cand)
    return before - after


def score_less_tracein(problem: dict, base_params: dict[str, np.ndarray]) -> np.ndarray:
    g_val = grad_mean(base_params, problem["x_val"], problem["y_val"])
    cand = problem["candidate_idx"]
    return grad_dot_per_sample(base_params, problem["x_train"][cand], problem["y_train"][cand], g_val)


def score_grad_norm(problem: dict, base_params: dict[str, np.ndarray]) -> np.ndarray:
    cand = problem["candidate_idx"]
    return grad_norm_per_sample(base_params, problem["x_train"][cand], problem["y_train"][cand])


def oracle_candidate_utilities(problem: dict, base_params: dict[str, np.ndarray], config: dict) -> dict[str, np.ndarray]:
    cand = problem["candidate_idx"]
    x_cand = problem["x_train"][cand]
    y_cand = problem["y_train"][cand]
    x_val = problem["x_val"]
    y_val = problem["y_val"]

    base_logits_val = x_val @ base_params["w"] + base_params["b"]
    base_val_loss = binary_losses_from_logits(base_logits_val, y_val).mean()

    err0 = sigmoid(x_cand @ base_params["w"] + base_params["b"]) - y_cand
    cross = x_cand @ x_val.T
    one_step_logits = base_logits_val[None, :] - config["oracle_lr"] * err0[:, None] * (cross + 1.0)
    one_step_val_loss = binary_losses_from_logits(one_step_logits, y_val[None, :]).mean(axis=1)

    w_many = np.repeat(base_params["w"][None, :], len(cand), axis=0)
    b_many = np.full(len(cand), float(base_params["b"]), dtype=np.float64)
    for lr in make_schedule(config["oracle_horizon_steps"], config["oracle_lr"]):
        err = sigmoid(np.einsum("nd,nd->n", w_many, x_cand) + b_many) - y_cand
        w_many -= lr * err[:, None] * x_cand
        b_many -= lr * err
    horizon_logits = w_many @ x_val.T + b_many[:, None]
    horizon_val_loss = binary_losses_from_logits(horizon_logits, y_val[None, :]).mean(axis=1)

    return {
        "base_val_loss": float(base_val_loss),
        "one_step_utility": base_val_loss - one_step_val_loss,
        "horizon_utility": base_val_loss - horizon_val_loss,
    }


def retrain_selected(problem: dict, selected_rel: np.ndarray, config: dict) -> dict[str, float]:
    selected_idx = problem["candidate_idx"][selected_rel]
    init_params = clone_params(problem["retrain_params0"])
    init_val_loss = mean_loss(init_params, problem["x_val"], problem["y_val"])
    params = train_steps(
        init_params,
        problem["x_train"][selected_idx],
        problem["y_train"][selected_idx],
        make_schedule(config["retrain_steps"], config["retrain_lr0"]),
    )
    final_val_loss = mean_loss(params, problem["x_val"], problem["y_val"])
    return {
        "target_test_acc": accuracy(params, problem["x_test"], problem["y_test"]),
        "target_test_loss": mean_loss(params, problem["x_test"], problem["y_test"]),
        "target_val_loss": final_val_loss,
        "target_val_loss_drop": init_val_loss - final_val_loss,
    }


def evaluate_method(problem: dict, scores: np.ndarray, oracle: dict, config: dict) -> dict:
    cand = problem["candidate_idx"]
    positive_mask = problem["env_train"][cand] == 0
    order = np.argsort(scores)[::-1]
    out = {
        "mechanism": {
            "auc_target_vs_other": float("nan") if positive_mask.sum() == 0 else binary_auc(scores, positive_mask),
            "spearman_oracle_one_step": spearman_rank_correlation(scores, oracle["one_step_utility"]),
            "spearman_oracle_horizon": spearman_rank_correlation(scores, oracle["horizon_utility"]),
            "k": {},
        },
        "performance": {
            "k": {},
        },
    }
    for k in config["k_values"]:
        k_eff = min(max(int(k), 1), len(cand))
        selected = order[:k_eff]
        mech_row = topk_stats(scores, positive_mask, k_eff)
        mech_row.update(
            {
                "oracle_one_step_mean": float(oracle["one_step_utility"][selected].mean()),
                "oracle_horizon_mean": float(oracle["horizon_utility"][selected].mean()),
            }
        )
        out["mechanism"]["k"][str(k)] = mech_row
        out["performance"]["k"][str(k)] = retrain_selected(problem, selected, config)
    return out


def aggregate(seed_results: list[dict], methods: list[str], k_values: list[int]) -> dict:
    summary: dict[str, dict] = {}
    for method in methods:
        mech = seed_results[0]["methods"][method]["mechanism"]
        perf = seed_results[0]["methods"][method]["performance"]
        summary[method] = {
            "mechanism": {
                "auc_target_vs_other_mean": mean_std(
                    r["methods"][method]["mechanism"]["auc_target_vs_other"] for r in seed_results
                )["mean"],
                "auc_target_vs_other_std": mean_std(
                    r["methods"][method]["mechanism"]["auc_target_vs_other"] for r in seed_results
                )["std"],
                "spearman_oracle_one_step_mean": mean_std(
                    r["methods"][method]["mechanism"]["spearman_oracle_one_step"] for r in seed_results
                )["mean"],
                "spearman_oracle_one_step_std": mean_std(
                    r["methods"][method]["mechanism"]["spearman_oracle_one_step"] for r in seed_results
                )["std"],
                "spearman_oracle_horizon_mean": mean_std(
                    r["methods"][method]["mechanism"]["spearman_oracle_horizon"] for r in seed_results
                )["mean"],
                "spearman_oracle_horizon_std": mean_std(
                    r["methods"][method]["mechanism"]["spearman_oracle_horizon"] for r in seed_results
                )["std"],
                "k": {},
            },
            "performance": {
                "k": {},
            },
        }
        for k in k_values:
            k_key = str(k)
            mech_row = {}
            for metric in ["purity", "recall", "oracle_one_step_mean", "oracle_horizon_mean"]:
                stats = mean_std(r["methods"][method]["mechanism"]["k"][k_key][metric] for r in seed_results)
                mech_row[f"{metric}_mean"] = stats["mean"]
                mech_row[f"{metric}_std"] = stats["std"]
            perf_row = {}
            for metric in ["target_test_acc", "target_test_loss", "target_val_loss", "target_val_loss_drop"]:
                stats = mean_std(r["methods"][method]["performance"]["k"][k_key][metric] for r in seed_results)
                perf_row[f"{metric}_mean"] = stats["mean"]
                perf_row[f"{metric}_std"] = stats["std"]
            summary[method]["mechanism"]["k"][k_key] = mech_row
            summary[method]["performance"]["k"][k_key] = perf_row
    return summary


def run(config: dict, out_path: Path) -> dict:
    start = time.time()
    seed_results = []
    base_lrs = make_schedule(config["base_steps"], config["base_lr0"])
    val_lrs = make_schedule(config["val_steps"], config["val_lr0"])
    for seed in config["seeds"]:
        print(f"\n=== seed {seed} ===", flush=True)
        problem = make_problem(config, seed)
        base_params = train_steps(
            problem["params0"],
            problem["x_train"][problem["base_idx"]],
            problem["y_train"][problem["base_idx"]],
            base_lrs,
        )
        cand = problem["candidate_idx"]
        rng = np.random.default_rng(seed + 999)
        scores = {
            "TACS": score_tacs(problem, base_params, config, val_lrs),
            "ToV_1step": score_tov_1step(problem, base_params, config, val_lrs),
            "LESS_tracein": score_less_tracein(problem, base_params),
            "GradNorm": score_grad_norm(problem, base_params),
            "Random": rng.random(len(cand)),
        }
        oracle = oracle_candidate_utilities(problem, base_params, config)
        payload = {
            "seed": seed,
            "candidate_count": int(len(cand)),
            "target_candidates": int((problem["env_train"][cand] == 0).sum()),
            "random_target_rate": float((problem["env_train"][cand] == 0).mean()),
            "base_target_rate": float((problem["env_train"][problem["base_idx"]] == 0).mean()),
            "oracle_base_val_loss": oracle["base_val_loss"],
            "methods": {},
        }
        for method in METHODS:
            payload["methods"][method] = evaluate_method(problem, scores[method], oracle, config)
            main_k = str(config["k_values"][-1])
            mech = payload["methods"][method]["mechanism"]
            perf = payload["methods"][method]["performance"]["k"][main_k]
            print(
                f"{method:>14} "
                f"rho_oracle={mech['spearman_oracle_horizon']:.3f} "
                f"purity@{main_k}={mech['k'][main_k]['purity']:.3f} "
                f"test_acc={perf['target_test_acc']:.3f}",
                flush=True,
            )
        seed_results.append(payload)

    result = {
        "experiment": "logistic_shift_framework",
        "methods": METHODS,
        "config": config,
        "elapsed_seconds": time.time() - start,
        "seed_results": seed_results,
        "summary": aggregate(seed_results, METHODS, config["k_values"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    main_k = str(config["k_values"][-1])
    for method in METHODS:
        mech = result["summary"][method]["mechanism"]
        perf = result["summary"][method]["performance"]["k"][main_k]
        print(
            f"{method:>14} "
            f"rho_oracle={mech['spearman_oracle_horizon_mean']:.3f}+-{mech['spearman_oracle_horizon_std']:.3f} "
            f"purity@{main_k}={mech['k'][main_k]['purity_mean']:.3f} "
            f"test_acc={perf['target_test_acc_mean']:.3f} "
            f"test_loss={perf['target_test_loss_mean']:.3f}",
            flush=True,
        )
    return result


def build_config(quick: bool) -> dict:
    config = {
        "dim": 48,
        "n_distractors": 6,
        "target_env_rate": 0.05,
        "train_size": 16000,
        "base_size": 2000,
        "val_size": 8,
        "test_size": 2000,
        "class_sep": 2.2,
        "distractor_angle_deg": 80.0,
        "env_offset_scale": 0.6,
        "noise_std": 1.0,
        "init_scale": 0.15,
        "base_steps": 36,
        "base_lr0": 0.55,
        "val_steps": 6,
        "val_lr0": 0.65,
        "val_lr_scale": 1.0,
        "retrain_steps": 120,
        "retrain_lr0": 0.7,
        "oracle_horizon_steps": 6,
        "oracle_lr": 0.35,
        "k_values": [50, 100, 200, 400],
        "score_eps": 1e-12,
        "seeds": [3, 7, 42, 123, 2026, 31415, 27182, 16180, 101, 202],
    }
    if quick:
        config.update(
            {
                "train_size": 8000,
                "base_size": 1000,
                "test_size": 1200,
                "base_steps": 28,
                "val_steps": 5,
                "retrain_steps": 80,
                "k_values": [50, 100, 200],
                "seeds": [3, 7, 42],
            }
        )
    return config


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Local TACS logistic-regression shift experiment.")
    ap.add_argument("--quick", action="store_true", help="Run a smaller, faster smoke-test configuration.")
    ap.add_argument("--output", type=Path, default=OUT_PATH, help="Where to write the JSON results.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run(build_config(args.quick), args.output)


if __name__ == "__main__":
    main()
