#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "experiments" / "tacs_local" / "logistic_shift"
sys.path.insert(0, str(LOG_DIR))

import plot_logistic_path_alignment as path_plot  # noqa: E402


METHOD_STYLE = {
    "TACS": {"label": "TACS", "color": "#c62828"},
    "ToV_max_improv": {"label": "ToV", "color": "#2ca02c"},
    "LESS_multi": {"label": "LESS", "color": "#1f77b4"},
    "Random": {"label": "Random", "color": "#7a7a7a"},
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "stix",
            "font.size": 5.9,
            "axes.titlesize": 6.4,
            "axes.labelsize": 5.9,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 5.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontsize=6.9, fontweight="bold", va="top")


def reconstruct_endpoint_displacements(result: dict, strategy: str, method: str, k: int, seed: int) -> dict[str, np.ndarray]:
    config = result["config"]
    problem = path_plot.exp.make_problem(config, seed)
    params0 = path_plot.exp.init_params(config["dim"], config["init_scale"], np.random.default_rng(seed + 11_111))
    base_lrs = path_plot.exp.make_linear_decay_schedule(config["train_epochs"], config["lr0"])
    base_traj = path_plot.exp.train_trajectory(
        params0,
        problem["x_train"][problem["base_idx"]],
        problem["y_train"][problem["base_idx"]],
        base_lrs,
    )

    hp_best = None
    for seed_payload in result["seed_results"]:
        if int(seed_payload["seed"]) == int(seed):
            hp_best = seed_payload["tacs_hp_search"]["best"]
            break
    if hp_best is None:
        raise ValueError(f"missing TACS hparam payload for seed {seed}")
    tacs_lrs = path_plot.exp.make_linear_decay_schedule(int(hp_best["steps"]), float(hp_best["lr0"]))
    val_traj = path_plot.exp.train_trajectory(params0, problem["x_val"], problem["y_val"], tacs_lrs)
    scores = path_plot.score_vector_for_method(method, config, problem, params0, base_traj, tacs_lrs)
    selected_idx = path_plot.exp.select_indices(
        method=method,
        strategy=strategy,
        scores=scores,
        problem=problem,
        k=int(k),
        seed=seed * 100_000 + int(k) * 10 + path_plot.exp.METHODS.index(method) * 1000 + path_plot.exp.STRATEGIES.index(strategy),
    )
    real_traj = path_plot.exp.train_trajectory(
        params0,
        problem["x_train"][selected_idx],
        problem["y_train"][selected_idx],
        base_lrs,
    )
    paths = {}
    for key, traj in {"val": val_traj, "base": base_traj, "real": real_traj}.items():
        pts = path_plot.trajectory_points(traj, params0)
        disp = pts[-1:] - pts[:1]
        denom = max(float(np.linalg.norm(disp[-1])), 1e-12)
        paths[key] = np.concatenate([np.zeros_like(disp), disp / denom], axis=0)
    return path_plot.project_semantic(paths)


def collect_projected_endpoint_displacements(analysis: dict, result: dict, strategy: str, method: str, k: int) -> dict[str, list[np.ndarray]]:
    rows = path_plot.seed_row_lookup(analysis, strategy, method, k)
    seeds = sorted({int(row["seed"]) for row in rows})
    bundles: dict[str, list[np.ndarray]] = {"val": [], "base": [], "real": []}
    for seed in seeds:
        projected = reconstruct_endpoint_displacements(result, strategy, method, k, seed)
        for key in bundles:
            bundles[key].append(projected[key])
    return bundles


def plot_path_panel(ax: plt.Axes, projected: dict[str, np.ndarray]) -> None:
    labels = {"val": "Val ref", "base": "Pool ref", "real": "Retrain"}
    for key in ["val", "base", "real"]:
        cfg = path_plot.REF_STYLE[key]
        for i, pts in enumerate(projected[key]):
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=cfg["color"],
                linestyle=cfg["linestyle"],
                linewidth=0.95 if key != "base" else 0.85,
                alpha=0.32 if key != "real" else 0.40,
                label=labels[key] if i == 0 else None,
            )
            ax.scatter(pts[-1, 0], pts[-1, 1], color=cfg["color"], s=11, marker="D", zorder=4, alpha=0.55)
    ax.scatter(0.0, 0.0, color="#111111", s=10, zorder=5)
    ax.axhline(0.0, color="#9a9a9a", linewidth=0.45, alpha=0.35)
    ax.axvline(0.0, color="#9a9a9a", linewidth=0.45, alpha=0.35)
    ax.set_title("Endpoint displacement")
    ax.set_xlabel("progress")
    ax.set_ylabel("deviation")
    ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.32)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, loc="lower right", handlelength=1.4, labelspacing=0.2, borderaxespad=0.1)
    panel_label(ax, "a")


def plot_shape_panel(ax: plt.Axes, analysis: dict, strategy: str) -> None:
    budgets = [1024, 2048, 4096, 8192]
    series = path_plot.aggregate_series(analysis, strategy, ["TACS", "Random"], budgets, metric_kind="shape_ratio")
    seed_values = path_plot.per_seed_metric_values(analysis, strategy, ["TACS", "Random"], budgets, metric_kind="shape_ratio")
    x = np.arange(len(budgets), dtype=float)
    offsets = {"TACS": -0.055, "Random": 0.055}
    for method in ["TACS", "Random"]:
        cfg = METHOD_STYLE[method]
        xs = x + offsets[method]
        for i, vals in enumerate(seed_values[method]):
            ax.scatter(np.full_like(vals, xs[i]), vals, color=cfg["color"], s=8, alpha=0.22, edgecolors="none")
        ax.plot(
            xs,
            series[method]["mean"],
            color=cfg["color"],
            marker="o",
            markersize=3.0,
            linewidth=1.55,
            markerfacecolor=cfg["color"] if method == "TACS" else "white",
            label=cfg["label"],
        )
    ax.axhline(1.0, color="#333333", linewidth=0.55, linestyle=":")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["1k", "2k", "4k", "8k"])
    ax.set_title("Closer to val path")
    ax.set_xlabel("subset size")
    ax.set_ylabel("shape ratio")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.55, alpha=0.32)
    ax.legend(
        frameon=False,
        loc="center right",
        bbox_to_anchor=(1.0, 0.55),
        handlelength=1.2,
        borderaxespad=0.2,
        labelspacing=0.18,
    )
    panel_label(ax, "b")


def plot_error_panel(ax: plt.Axes, result: dict, strategy: str) -> None:
    budgets = [512, 1024, 2048, 4096, 8192]
    summary = result["summary"]
    for method in ["TACS", "ToV_max_improv", "LESS_multi"]:
        cfg = METHOD_STYLE[method]
        means = np.asarray([summary[strategy][method]["k"][str(k)]["classification_error_mean"] for k in budgets])
        stds = np.asarray([summary[strategy][method]["k"][str(k)]["classification_error_std"] for k in budgets])
        ax.plot(
            budgets,
            means,
            color=cfg["color"],
            marker="o",
            markersize=3.0,
            linewidth=1.65 if method == "TACS" else 1.35,
            markerfacecolor=cfg["color"] if method == "TACS" else "white",
            label=cfg["label"],
        )
        ax.fill_between(budgets, means - stds, means + stds, color=cfg["color"], alpha=0.10, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels(["512", "1k", "2k", "4k", "8k"])
    ax.set_title("Lower target error")
    ax.set_xlabel("subset size")
    ax.set_ylabel("classification error")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.55, alpha=0.32)
    ax.legend(frameon=False, loc="upper right", ncol=1, handlelength=1.2, borderaxespad=0.1, labelspacing=0.18)
    panel_label(ax, "c")


def main() -> None:
    style()
    result = load_json(LOG_DIR / "logistic_tov_style_comparison_full_hpsearch_ckpt1.json")
    analysis = load_json(LOG_DIR / "logistic_path_alignment_full_hpsearch_ckpt1.json")
    strategy = "score_only"
    projected = collect_projected_endpoint_displacements(analysis, result, strategy, "TACS", 8192)

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.78), gridspec_kw={"wspace": 0.34})
    plot_path_panel(axes[0], projected)
    plot_shape_panel(axes[1], analysis, strategy)
    plot_error_panel(axes[2], result, strategy)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.23, top=0.86)

    out = REPO / "Validation_Warmup" / "fig" / "toy_logistic_main_compact.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(out)


if __name__ == "__main__":
    main()
