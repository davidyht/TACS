#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_logistic_tov_style_comparison as exp


MAIN_METHODS = ["TACS", "ToV_max_improv", "LESS_multi", "Random"]
MAIN_STRATEGIES = ["score_plus_random", "score_only"]


def flatten_params(params: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([params["w"].reshape(-1), np.array([params["b"]], dtype=np.float64)])


def trajectory_points(traj: list[dict[str, np.ndarray]], params0: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([flatten_params(params0)] + [flatten_params(p) for p in traj], axis=0)


def resample_path(points: np.ndarray, n_points: int) -> np.ndarray:
    seg = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-12:
        return np.repeat(points[:1], n_points, axis=0)

    targets = np.linspace(0.0, total, n_points)
    out = []
    j = 0
    for target in targets:
        while j + 1 < len(cum) and cum[j + 1] < target:
            j += 1
        if j + 1 == len(cum):
            out.append(points[-1])
            continue
        lo, hi = cum[j], cum[j + 1]
        alpha = 0.0 if hi - lo <= 1e-12 else (target - lo) / (hi - lo)
        out.append((1.0 - alpha) * points[j] + alpha * points[j + 1])
    return np.stack(out, axis=0)


def normalize_shape(points: np.ndarray) -> np.ndarray:
    centered = points - points[:1]
    end_norm = np.linalg.norm(centered[-1])
    if end_norm <= 1e-12:
        return centered
    return centered / end_norm


def path_metrics(a_points: np.ndarray, b_points: np.ndarray, n_points: int) -> dict[str, float]:
    a = resample_path(a_points, n_points)
    b = resample_path(b_points, n_points)
    diff = a - b

    a_disp = a - a[:1]
    b_disp = b - b[:1]
    progress_cos = []
    for u, v in zip(a_disp[1:], b_disp[1:]):
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu <= 1e-12 or nv <= 1e-12:
            continue
        progress_cos.append(float(np.dot(u, v) / (nu * nv)))

    end_u = a_points[-1] - a_points[0]
    end_v = b_points[-1] - b_points[0]
    end_un = np.linalg.norm(end_u)
    end_vn = np.linalg.norm(end_v)
    end_cos = float("nan")
    if end_un > 1e-12 and end_vn > 1e-12:
        end_cos = float(np.dot(end_u, end_v) / (end_un * end_vn))

    shape_a = normalize_shape(a)
    shape_b = normalize_shape(b)
    shape_mean_l2 = float(np.linalg.norm(shape_a - shape_b, axis=1).mean())

    return {
        "raw_mean_l2": float(np.linalg.norm(diff, axis=1).mean()),
        "raw_end_l2": float(np.linalg.norm(a[-1] - b[-1])),
        "shape_mean_l2": shape_mean_l2,
        "progress_cos": float(np.mean(progress_cos)) if progress_cos else float("nan"),
        "end_cos": end_cos,
    }


def compute_scores(
    method: str,
    config: dict,
    problem: dict,
    params0: dict[str, np.ndarray],
    base_traj: list[dict[str, np.ndarray]],
    tacs_lrs: list[float],
) -> np.ndarray | None:
    if method == "Random":
        return None
    if method.startswith("ToV_"):
        tov_lrs = [config["val_lr_scale"] * lr for lr in exp.make_linear_decay_schedule(config["train_epochs"], config["lr0"])]
        return exp.score_tov_family(base_traj, problem, tov_lrs)[method]
    if method == "LESS_multi":
        return exp.score_less_multi(base_traj, problem)
    if method == "TACS":
        return exp.score_tacs_from_base_val_trajectory(params0, problem, tacs_lrs, config["tacs_score_window"])
    raise ValueError(f"unsupported method: {method}")


def analyze(result_json: Path, n_points: int) -> dict:
    result = json.loads(result_json.read_text())
    config = result["config"]
    base_lrs = exp.make_linear_decay_schedule(config["train_epochs"], config["lr0"])
    budgets = config["budgets"]

    rows = []
    for seed_payload in result["seed_results"]:
        seed = int(seed_payload["seed"])
        problem = exp.make_problem(config, seed)
        params0 = exp.init_params(config["dim"], config["init_scale"], np.random.default_rng(seed + 11_111))

        base_traj = exp.train_trajectory(
            params0,
            problem["x_train"][problem["base_idx"]],
            problem["y_train"][problem["base_idx"]],
            base_lrs,
        )
        base_points = trajectory_points(base_traj, params0)

        hp_best = seed_payload["tacs_hp_search"]["best"]
        tacs_lrs = exp.make_linear_decay_schedule(int(hp_best["steps"]), float(hp_best["lr0"]))
        val_traj = exp.train_trajectory(params0, problem["x_val"], problem["y_val"], tacs_lrs)
        val_points = trajectory_points(val_traj, params0)

        score_cache = {
            method: compute_scores(method, config, problem, params0, base_traj, tacs_lrs)
            for method in MAIN_METHODS
        }

        for strategy in MAIN_STRATEGIES:
            for method in MAIN_METHODS:
                method_idx = exp.METHODS.index(method)
                strategy_idx = exp.STRATEGIES.index(strategy)
                scores = score_cache[method]
                for k in budgets:
                    select_seed = seed * 100_000 + int(k) * 10 + method_idx * 1000 + strategy_idx
                    selected_idx = exp.select_indices(
                        method=method,
                        strategy=strategy,
                        scores=scores,
                        problem=problem,
                        k=int(k),
                        seed=select_seed,
                    )
                    final_traj = exp.train_trajectory(
                        params0,
                        problem["x_train"][selected_idx],
                        problem["y_train"][selected_idx],
                        base_lrs,
                    )
                    final_points = trajectory_points(final_traj, params0)
                    val_metrics = path_metrics(final_points, val_points, n_points)
                    base_metrics = path_metrics(final_points, base_points, n_points)
                    rows.append(
                        {
                            "seed": seed,
                            "strategy": strategy,
                            "method": method,
                            "k": int(k),
                            "val": val_metrics,
                            "base": base_metrics,
                            "val_better": {
                                "raw_mean_l2": val_metrics["raw_mean_l2"] < base_metrics["raw_mean_l2"],
                                "raw_end_l2": val_metrics["raw_end_l2"] < base_metrics["raw_end_l2"],
                                "shape_mean_l2": val_metrics["shape_mean_l2"] < base_metrics["shape_mean_l2"],
                                "progress_cos": val_metrics["progress_cos"] > base_metrics["progress_cos"],
                                "end_cos": val_metrics["end_cos"] > base_metrics["end_cos"],
                            },
                        }
                    )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["strategy"], row["method"], row["k"])].append(row)

    summary = {}
    for key, group in grouped.items():
        strategy, method, k = key
        out = {"n_seeds": len(group)}
        for family in ["val", "base"]:
            for metric in ["raw_mean_l2", "raw_end_l2", "shape_mean_l2", "progress_cos", "end_cos"]:
                vals = [float(row[family][metric]) for row in group]
                out[f"{family}_{metric}_mean"] = float(np.mean(vals))
                out[f"{family}_{metric}_std"] = float(np.std(vals, ddof=0))
        for metric in ["raw_mean_l2", "raw_end_l2", "shape_mean_l2", "progress_cos", "end_cos"]:
            vals = [float(row["val_better"][metric]) for row in group]
            out[f"val_better_{metric}_frac"] = float(np.mean(vals))
        summary.setdefault(strategy, {}).setdefault(method, {})[str(k)] = out

    return {
        "source_json": str(result_json.resolve()),
        "n_points": int(n_points),
        "budgets": [int(k) for k in budgets],
        "rows": rows,
        "summary": summary,
    }


def build_report(analysis: dict, main_budgets: list[int] | None) -> str:
    lines = []
    lines.append(f"Source: {analysis['source_json']}")
    lines.append("")
    for strategy in MAIN_STRATEGIES:
        if strategy not in analysis["summary"]:
            continue
        lines.append(f"[{strategy}]")
        for method in MAIN_METHODS:
            method_rows = analysis["summary"][strategy].get(method, {})
            if not method_rows:
                continue
            lines.append(method)
            budgets = sorted(int(k) for k in method_rows.keys())
            if main_budgets:
                budgets = [k for k in budgets if k in main_budgets]
            for k in budgets:
                row = method_rows[str(k)]
                lines.append(
                    "  "
                    f"k={k}: "
                    f"raw_l2 val/base={row['val_raw_mean_l2_mean']:.4f}/{row['base_raw_mean_l2_mean']:.4f}, "
                    f"shape_l2 val/base={row['val_shape_mean_l2_mean']:.4f}/{row['base_shape_mean_l2_mean']:.4f}, "
                    f"progress_cos val/base={row['val_progress_cos_mean']:.4f}/{row['base_progress_cos_mean']:.4f}, "
                    f"end_cos val/base={row['val_end_cos_mean']:.4f}/{row['base_end_cos_mean']:.4f}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze logistic-regression trajectory alignment versus TACS val warmup and pool warmup.")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().with_name("logistic_tov_style_comparison_full_hpsearch_ckpt1.json"),
        help="Source logistic result JSON.",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().with_name("logistic_path_alignment_full_hpsearch_ckpt1.json"),
        help="Where to write the detailed alignment JSON.",
    )
    ap.add_argument(
        "--output-report",
        type=Path,
        default=Path(__file__).resolve().with_name("logistic_path_alignment_full_hpsearch_ckpt1.txt"),
        help="Where to write the text summary.",
    )
    ap.add_argument("--resample-points", type=int, default=64)
    ap.add_argument("--main-budgets", nargs="*", type=int, default=[1024, 2048, 4096, 8192])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    analysis = analyze(args.input, args.resample_points)
    args.output_json.write_text(json.dumps(analysis, indent=2))
    args.output_report.write_text(build_report(analysis, args.main_budgets))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_report}")


if __name__ == "__main__":
    main()
