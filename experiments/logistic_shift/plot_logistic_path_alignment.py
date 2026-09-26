#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_logistic_tov_style_comparison as exp


METHODS = ["TACS", "ToV_max_improv", "LESS_multi", "Random"]
MAIN_METHODS = ["TACS", "Random"]
METHOD_STYLE = {
    "TACS": {"label": "TACS", "color": "#c62828"},
    "ToV_max_improv": {"label": "ToV", "color": "#2ca02c"},
    "LESS_multi": {"label": "LESS", "color": "#1f77b4"},
    "Random": {"label": "Random", "color": "#2b6ea6"},
}
REF_STYLE = {
    "val": {"label": "Val warmup reference", "color": "#c62828", "linestyle": "-"},
    "base": {"label": "Pool warmup reference", "color": "#4d4d4d", "linestyle": "--"},
    "real": {"label": "Real retrain path", "color": "#111111", "linestyle": "-"},
}
PANEL_LABELS = ["A", "B", "C", "D"]
SINGLE_PANEL_SIZE = (3.2, 2.8)


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_compact_axis(ax: plt.Axes) -> None:
    ax.title.set_fontsize(9.8)
    ax.xaxis.label.set_size(8.8)
    ax.yaxis.label.set_size(8.8)
    ax.tick_params(axis="both", labelsize=7.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def add_panel_label(ax: plt.Axes, label: str | None) -> None:
    if not label:
        return
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top",
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_params(params: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([params["w"].reshape(-1), np.array([params["b"]], dtype=np.float64)])


def trajectory_points(traj: list[dict[str, np.ndarray]], params0: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([flatten_params(params0)] + [flatten_params(p) for p in traj], axis=0)


def resample_path(points: np.ndarray, n_points: int = 64) -> np.ndarray:
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


def project_pca(*paths: np.ndarray) -> list[np.ndarray]:
    all_points = np.concatenate(paths, axis=0)
    mean = all_points.mean(axis=0, keepdims=True)
    centered = all_points - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    return [(path - mean) @ basis for path in paths]


def semantic_projection_basis(paths: dict[str, np.ndarray]) -> np.ndarray:
    eps = 1e-12
    real_end = paths["real"][-1]
    real_norm = np.linalg.norm(real_end)
    if real_norm <= eps:
        raise ValueError("real path endpoint has near-zero norm")
    e1 = real_end / real_norm

    val_end = paths["val"][-1]
    val_orth = val_end - np.dot(val_end, e1) * e1
    val_orth_norm = np.linalg.norm(val_orth)
    if val_orth_norm <= eps:
        base_end = paths["base"][-1]
        val_orth = base_end - np.dot(base_end, e1) * e1
        val_orth_norm = np.linalg.norm(val_orth)
    if val_orth_norm <= eps:
        axis = np.zeros_like(e1)
        axis[int(np.argmin(np.abs(e1)))] = 1.0
        val_orth = axis - np.dot(axis, e1) * e1
        val_orth_norm = np.linalg.norm(val_orth)
    e2 = val_orth / max(val_orth_norm, eps)
    if np.dot(val_end, e2) < 0.0:
        e2 = -e2
    return np.stack([e1, e2], axis=1)


def project_semantic(paths: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    basis = semantic_projection_basis(paths)
    return {key: value @ basis for key, value in paths.items()}


def score_vector_for_method(
    method: str,
    config: dict,
    problem: dict,
    params0: dict[str, np.ndarray],
    base_traj: list[dict[str, np.ndarray]],
    tacs_lrs: list[float],
) -> np.ndarray | None:
    if method == "Random":
        return None
    if method == "TACS":
        return exp.score_tacs_from_base_val_trajectory(params0, problem, tacs_lrs, config["tacs_score_window"])
    if method == "LESS_multi":
        return exp.score_less_multi(base_traj, problem)
    tov_lrs = [config["val_lr_scale"] * lr for lr in exp.make_linear_decay_schedule(config["train_epochs"], config["lr0"])]
    return exp.score_tov_family(base_traj, problem, tov_lrs)[method]


def seed_row_lookup(analysis: dict, strategy: str, method: str, k: int) -> list[dict]:
    return [row for row in analysis["rows"] if row["strategy"] == strategy and row["method"] == method and int(row["k"]) == int(k)]


def choose_representative_seed(analysis: dict, strategy: str, method: str, k: int) -> int:
    rows = seed_row_lookup(analysis, strategy, method, k)
    ratios = np.asarray([row["val"]["shape_mean_l2"] / max(row["base"]["shape_mean_l2"], 1e-12) for row in rows], dtype=np.float64)
    target = float(np.median(ratios))
    best_idx = int(np.argmin(np.abs(ratios - target)))
    return int(rows[best_idx]["seed"])


def reconstruct_paths(
    result: dict,
    strategy: str,
    method: str,
    k: int,
    seed: int,
) -> dict[str, np.ndarray]:
    config = result["config"]
    problem = exp.make_problem(config, seed)
    params0 = exp.init_params(config["dim"], config["init_scale"], np.random.default_rng(seed + 11_111))
    base_lrs = exp.make_linear_decay_schedule(config["train_epochs"], config["lr0"])
    base_traj = exp.train_trajectory(
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
    assert hp_best is not None
    tacs_lrs = exp.make_linear_decay_schedule(int(hp_best["steps"]), float(hp_best["lr0"]))
    val_traj = exp.train_trajectory(params0, problem["x_val"], problem["y_val"], tacs_lrs)

    scores = score_vector_for_method(method, config, problem, params0, base_traj, tacs_lrs)
    selected_idx = exp.select_indices(
        method=method,
        strategy=strategy,
        scores=scores,
        problem=problem,
        k=int(k),
        seed=seed * 100_000 + int(k) * 10 + exp.METHODS.index(method) * 1000 + exp.STRATEGIES.index(strategy),
    )
    real_traj = exp.train_trajectory(
        params0,
        problem["x_train"][selected_idx],
        problem["y_train"][selected_idx],
        base_lrs,
    )

    return {
        "val": normalize_shape(resample_path(trajectory_points(val_traj, params0))),
        "base": normalize_shape(resample_path(trajectory_points(base_traj, params0))),
        "real": normalize_shape(resample_path(trajectory_points(real_traj, params0))),
    }


def collect_projected_paths(
    analysis: dict,
    result: dict,
    strategy: str,
    method: str,
    k: int,
) -> dict[str, np.ndarray]:
    rows = seed_row_lookup(analysis, strategy, method, k)
    seeds = sorted({int(row["seed"]) for row in rows})
    bundles = {"val": [], "base": [], "real": []}
    for seed in seeds:
        paths = reconstruct_paths(result, strategy, method, k, seed)
        projected = project_semantic(paths)
        for key in bundles:
            bundles[key].append(projected[key])
    return {key: np.stack(value, axis=0) for key, value in bundles.items()}


def plot_overlay(ax: plt.Axes, paths: dict[str, np.ndarray], title: str, panel_label: str) -> None:
    val_2d, base_2d, real_2d = project_pca(paths["val"], paths["base"], paths["real"])
    proj = {"val": val_2d, "base": base_2d, "real": real_2d}

    for key in ["val", "base", "real"]:
        style = REF_STYLE[key]
        pts = proj[key]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.4 if key != "base" else 2.0,
            alpha=0.95,
            label=style["label"],
        )
        ax.scatter(pts[0, 0], pts[0, 1], color=style["color"], s=20, zorder=3)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=style["color"], s=30, marker="D", zorder=3)

    ax.annotate("Init", (real_2d[0, 0], real_2d[0, 1]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, linestyle=":", linewidth=0.9, alpha=0.35, color="#b8b8b8")
    ax.set_aspect("equal", adjustable="datalim")
    add_panel_label(ax, panel_label)


def plot_overlay_all_seeds(
    ax: plt.Axes,
    projected_paths: dict[str, np.ndarray],
    title: str,
    panel_label: str,
) -> None:
    compact = ax.figure.get_size_inches()[0] <= 3.3
    for key in ["val", "base", "real"]:
        style = REF_STYLE[key]
        stack = projected_paths[key]
        for pts in stack:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.0,
                alpha=0.12 if key != "real" else 0.10,
                zorder=1,
            )
        mean_pts = stack.mean(axis=0)
        ax.plot(
            mean_pts[:, 0],
            mean_pts[:, 1],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.8 if key != "base" else 2.3,
            alpha=0.98,
            label=style["label"],
            zorder=3,
        )
        ax.scatter(stack[:, 0, 0], stack[:, 0, 1], color=style["color"], s=12, alpha=0.16, zorder=2)
        ax.scatter(stack[:, -1, 0], stack[:, -1, 1], color=style["color"], s=18, alpha=0.25, zorder=2)
        ax.scatter(mean_pts[-1, 0], mean_pts[-1, 1], color=style["color"], s=34, marker="D", zorder=4)

    ax.scatter(0.0, 0.0, color="#111111", s=24, zorder=5)
    if not compact:
        ax.annotate("Init", (0.0, 0.0), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Progress Along Selected Path" if not compact else "Progress Along Path")
    ax.set_ylabel("Deviation toward val path" if not compact else "Deviation Toward Val")
    ax.grid(True, linestyle=":", linewidth=0.9, alpha=0.35, color="#b8b8b8")
    ax.axhline(0.0, color="#8a8a8a", linewidth=0.8, alpha=0.25, zorder=0)
    ax.axvline(0.0, color="#8a8a8a", linewidth=0.8, alpha=0.25, zorder=0)
    ax.set_aspect("equal", adjustable="datalim")
    add_panel_label(ax, panel_label)

    if not compact:
        label_specs = [
            ("val", "Val warmup", (-6, 6), "right"),
            ("base", "Pool warmup", (-6, -10), "right"),
            ("real", "Real path", (-6, -2), "right"),
        ]
        for key, label, offset, ha in label_specs:
            mean_pts = projected_paths[key].mean(axis=0)
            style = REF_STYLE[key]
            ax.annotate(
                label,
                (mean_pts[-1, 0], mean_pts[-1, 1]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.8,
                color=style["color"],
                ha=ha,
                va="center",
            )
    ax.margins(x=0.08, y=0.10)


def aggregate_series(analysis: dict, strategy: str, methods: list[str], budgets: list[int], metric_kind: str) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for method in methods:
        means = []
        stds = []
        for k in budgets:
            rows = seed_row_lookup(analysis, strategy, method, k)
            if metric_kind == "shape_ratio":
                vals = np.asarray(
                    [row["val"]["shape_mean_l2"] / max(row["base"]["shape_mean_l2"], 1e-12) for row in rows],
                    dtype=np.float64,
                )
            elif metric_kind == "end_cos_gap":
                vals = np.asarray(
                    [row["val"]["end_cos"] - row["base"]["end_cos"] for row in rows],
                    dtype=np.float64,
                )
            else:
                raise ValueError(metric_kind)
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=0)))
        out[method] = {
            "mean": np.asarray(means, dtype=np.float64),
            "std": np.asarray(stds, dtype=np.float64),
        }
    return out


def per_seed_metric_values(
    analysis: dict,
    strategy: str,
    methods: list[str],
    budgets: list[int],
    metric_kind: str,
) -> dict[str, list[np.ndarray]]:
    out = {}
    for method in methods:
        by_budget = []
        for k in budgets:
            rows = seed_row_lookup(analysis, strategy, method, k)
            if metric_kind == "shape_ratio":
                vals = np.asarray(
                    [row["val"]["shape_mean_l2"] / max(row["base"]["shape_mean_l2"], 1e-12) for row in rows],
                    dtype=np.float64,
                )
            elif metric_kind == "end_cos_gap":
                vals = np.asarray(
                    [row["val"]["end_cos"] - row["base"]["end_cos"] for row in rows],
                    dtype=np.float64,
                )
            else:
                raise ValueError(metric_kind)
            by_budget.append(vals)
        out[method] = by_budget
    return out


def plot_metric_panel(
    ax: plt.Axes,
    budgets: list[int],
    series: dict[str, dict[str, np.ndarray]],
    metric_kind: str,
    panel_label: str,
    methods: list[str],
) -> None:
    for method in methods:
        style = METHOD_STYLE[method]
        mean = series[method]["mean"]
        std = series[method]["std"]
        ax.plot(
            budgets,
            mean,
            color=style["color"],
            linewidth=2.4 if method == "TACS" else 2.0,
            marker="o",
            markersize=5.2,
            markerfacecolor="white" if method != "TACS" else style["color"],
            markeredgewidth=1.1,
            label=style["label"],
        )
        ax.fill_between(budgets, mean - std, mean + std, color=style["color"], alpha=0.10)

    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels(["1k", "2k", "4k", "8k"])
    ax.set_xlabel("Final subset size n")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.9, alpha=0.35, color="#b8b8b8")

    if metric_kind == "shape_ratio":
        ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle=":")
        ax.set_yscale("log")
        ax.set_ylabel("Shape distance ratio (val / base)")
        ax.set_title("Shape Similarity to Reference")
    else:
        ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle=":")
        ax.set_ylabel("Endpoint cosine gap (val - base)")
        ax.set_title("Endpoint Direction Similarity")

    add_panel_label(ax, panel_label)


def plot_metric_panel_all_seeds(
    ax: plt.Axes,
    budgets: list[int],
    series: dict[str, dict[str, np.ndarray]],
    seed_values: dict[str, list[np.ndarray]],
    metric_kind: str,
    panel_label: str,
    methods: list[str],
) -> None:
    xlocs = np.arange(len(budgets), dtype=np.float64)
    offsets = np.linspace(-0.10, 0.10, len(methods))
    compact = ax.figure.get_size_inches()[0] <= 3.3

    for offset, method in zip(offsets, methods):
        style = METHOD_STYLE[method]
        mean = series[method]["mean"]
        xs = xlocs + offset
        for i, vals in enumerate(seed_values[method]):
            if len(vals) == 0:
                continue
            jitter = np.linspace(-0.03, 0.03, len(vals))
            ax.scatter(
                np.full(len(vals), xs[i]) + jitter,
                vals,
                color=style["color"],
                s=26,
                alpha=0.22,
                edgecolors="none",
                zorder=1,
            )
        ax.plot(
            xs,
            mean,
            color=style["color"],
            linewidth=2.4 if method == "TACS" else 2.0,
            marker="o",
            markersize=5.4,
            markerfacecolor="white" if method != "TACS" else style["color"],
            markeredgewidth=1.1,
            label=style["label"],
            zorder=3,
        )

    ax.set_xlim(-0.45, len(budgets) - 0.55)
    ax.set_xticks(xlocs)
    ax.set_xticklabels(["1k", "2k", "4k", "8k"])
    ax.set_xlabel("Final subset size n")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.9, alpha=0.35, color="#b8b8b8")

    if metric_kind == "shape_ratio":
        ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle=":")
        ax.set_yscale("log")
        ax.set_ylabel("Shape distance ratio (val / base)" if not compact else "Shape ratio (val/base)")
        ax.set_title("Shape Similarity to Reference")
    else:
        ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle=":")
        ax.set_ylabel("Endpoint cosine gap (val - base)" if not compact else "Endpoint cosine gap")
        ax.set_title("Endpoint Direction Similarity")

    add_panel_label(ax, panel_label)

    if not compact:
        for method in methods:
            style = METHOD_STYLE[method]
            mean = series[method]["mean"]
            x = xlocs[-1] + offsets[methods.index(method)]
            y = mean[-1]
            ax.annotate(
                style["label"],
                (x, y),
                xytext=(-6, 0 if method == "Random" else 4),
                textcoords="offset points",
                fontsize=10,
                color=style["color"],
                ha="right",
                va="center",
            )


def save_panel(fig: plt.Figure, out_path: Path, *, tight: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches=bbox)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches=bbox)
    plt.close(fig)


def build_figure(result: dict, analysis: dict, out_prefix: Path, strategy: str, overlay_budget: int) -> None:
    style_matplotlib()
    budgets = [int(k) for k in analysis["budgets"] if int(k) in {1024, 2048, 4096, 8192}]

    rep_seed_tacs = choose_representative_seed(analysis, strategy, "TACS", overlay_budget)
    rep_seed_random = choose_representative_seed(analysis, strategy, "Random", overlay_budget)
    tacs_paths = reconstruct_paths(result, strategy, "TACS", overlay_budget, rep_seed_tacs)
    random_paths = reconstruct_paths(result, strategy, "Random", overlay_budget, rep_seed_random)

    shape_series = aggregate_series(analysis, strategy, MAIN_METHODS, budgets, metric_kind="shape_ratio")
    end_cos_series = aggregate_series(analysis, strategy, MAIN_METHODS, budgets, metric_kind="end_cos_gap")

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    plot_overlay(
        axes[0, 0],
        tacs_paths,
        title=f"TACS-selected subset vs references ({strategy}, k={overlay_budget}, seed={rep_seed_tacs})",
        panel_label=PANEL_LABELS[0],
    )
    plot_overlay(
        axes[0, 1],
        random_paths,
        title=f"Random-selected subset vs references ({strategy}, k={overlay_budget}, seed={rep_seed_random})",
        panel_label=PANEL_LABELS[1],
    )
    plot_metric_panel(axes[1, 0], budgets, shape_series, "shape_ratio", PANEL_LABELS[2], MAIN_METHODS)
    plot_metric_panel(axes[1, 1], budgets, end_cos_series, "end_cos_gap", PANEL_LABELS[3], MAIN_METHODS)

    overlay_handles, overlay_labels = axes[0, 0].get_legend_handles_labels()
    metric_handles, metric_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        overlay_handles + metric_handles,
        overlay_labels + metric_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.suptitle(
        "Validation warmup is a better reference path for selected-subset dynamics in logistic regression",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    save_panel(fig, out_prefix)

    panel_specs = [
        ("panel_A_tacs_path", lambda ax: plot_overlay(
            ax,
            tacs_paths,
            title=f"TACS-selected subset vs references ({strategy}, k={overlay_budget}, seed={rep_seed_tacs})",
            panel_label="",
        )),
        ("panel_B_random_path", lambda ax: plot_overlay(
            ax,
            random_paths,
            title=f"Random-selected subset vs references ({strategy}, k={overlay_budget}, seed={rep_seed_random})",
            panel_label="",
        )),
        ("panel_C_shape_ratio", lambda ax: plot_metric_panel(
            ax, budgets, shape_series, "shape_ratio", "", MAIN_METHODS
        )),
        ("panel_D_end_cos_gap", lambda ax: plot_metric_panel(
            ax, budgets, end_cos_series, "end_cos_gap", "", MAIN_METHODS
        )),
    ]
    for name, draw in panel_specs:
        single_fig, single_ax = plt.subplots(1, 1, figsize=(5.9, 4.7))
        draw(single_ax)
        if name in {"panel_A_tacs_path", "panel_B_random_path"}:
            handles, labels = single_ax.get_legend_handles_labels()
            single_fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=3,
                frameon=False,
                columnspacing=1.0,
                handletextpad=0.5,
            )
            single_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        else:
            handles, labels = single_ax.get_legend_handles_labels()
            single_fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=2,
                frameon=False,
                columnspacing=1.0,
                handletextpad=0.5,
            )
            single_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        save_panel(single_fig, out_prefix.parent / f"{out_prefix.name}_{name}")


def build_aggregate_figure(result: dict, analysis: dict, out_prefix: Path, strategy: str, overlay_budget: int) -> None:
    style_matplotlib()
    budgets = [int(k) for k in analysis["budgets"] if int(k) in {1024, 2048, 4096, 8192}]

    tacs_projected = collect_projected_paths(analysis, result, strategy, "TACS", overlay_budget)
    random_projected = collect_projected_paths(analysis, result, strategy, "Random", overlay_budget)
    shape_series = aggregate_series(analysis, strategy, MAIN_METHODS, budgets, metric_kind="shape_ratio")
    end_cos_series = aggregate_series(analysis, strategy, MAIN_METHODS, budgets, metric_kind="end_cos_gap")
    shape_seed_values = per_seed_metric_values(analysis, strategy, MAIN_METHODS, budgets, metric_kind="shape_ratio")
    end_cos_seed_values = per_seed_metric_values(analysis, strategy, MAIN_METHODS, budgets, metric_kind="end_cos_gap")

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    plot_overlay_all_seeds(
        axes[0, 0],
        tacs_projected,
        title="TACS-Selected Subset",
        panel_label=PANEL_LABELS[0],
    )
    plot_overlay_all_seeds(
        axes[0, 1],
        random_projected,
        title="Random-Selected Subset",
        panel_label=PANEL_LABELS[1],
    )
    plot_metric_panel_all_seeds(
        axes[1, 0],
        budgets,
        shape_series,
        shape_seed_values,
        "shape_ratio",
        PANEL_LABELS[2],
        MAIN_METHODS,
    )
    plot_metric_panel_all_seeds(
        axes[1, 1],
        budgets,
        end_cos_series,
        end_cos_seed_values,
        "end_cos_gap",
        PANEL_LABELS[3],
        MAIN_METHODS,
    )

    overlay_handles, overlay_labels = axes[0, 0].get_legend_handles_labels()
    metric_handles, metric_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        overlay_handles + metric_handles,
        overlay_labels + metric_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.suptitle(
        "Validation Warmup Better Matches Selected-Subset Dynamics",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    save_panel(fig, out_prefix)

    panel_specs = [
        ("panel_A_tacs_path", lambda ax: plot_overlay_all_seeds(
            ax,
            tacs_projected,
            title="TACS Subset",
            panel_label="",
        )),
        ("panel_B_random_path", lambda ax: plot_overlay_all_seeds(
            ax,
            random_projected,
            title="Random Subset",
            panel_label="",
        )),
        ("panel_C_shape_ratio", lambda ax: plot_metric_panel_all_seeds(
            ax, budgets, shape_series, shape_seed_values, "shape_ratio", "", MAIN_METHODS
        )),
        ("panel_D_end_cos_gap", lambda ax: plot_metric_panel_all_seeds(
            ax, budgets, end_cos_series, end_cos_seed_values, "end_cos_gap", "", MAIN_METHODS
        )),
    ]
    for name, draw in panel_specs:
        single_fig, single_ax = plt.subplots(1, 1, figsize=SINGLE_PANEL_SIZE)
        draw(single_ax)
        style_compact_axis(single_ax)
        handles, labels = single_ax.get_legend_handles_labels()
        if name in {"panel_A_tacs_path", "panel_B_random_path"}:
            labels = ["Val ref", "Pool ref", "Real path"]
        single_ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=6.6 if name in {"panel_A_tacs_path", "panel_B_random_path"} else 7.0,
            handlelength=1.8,
            borderaxespad=0.0,
            labelspacing=0.35,
        )
        single_fig.subplots_adjust(left=0.23, right=0.68, bottom=0.22, top=0.84)
        save_panel(single_fig, out_prefix.parent / f"{out_prefix.name}_{name}", tight=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot motivation figure for logistic path alignment.")
    ap.add_argument(
        "--result-json",
        type=Path,
        default=Path(__file__).resolve().with_name("logistic_tov_style_comparison_full_hpsearch_ckpt1.json"),
    )
    ap.add_argument(
        "--analysis-json",
        type=Path,
        default=Path(__file__).resolve().with_name("logistic_path_alignment_full_hpsearch_ckpt1.json"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().with_name("plots_path_alignment"),
    )
    ap.add_argument("--strategy", choices=exp.STRATEGIES, default="score_plus_random")
    ap.add_argument("--overlay-budget", type=int, default=8192)
    ap.add_argument("--figure-kind", choices=["representative", "aggregate", "both"], default="aggregate")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    result = load_json(args.result_json)
    analysis = load_json(args.analysis_json)
    if args.figure_kind in {"representative", "both"}:
        out_prefix = args.out_dir / f"logistic_path_alignment_{args.strategy}"
        build_figure(result, analysis, out_prefix, args.strategy, args.overlay_budget)
        print(f"wrote {out_prefix.with_suffix('.pdf')}")
        print(f"wrote {out_prefix.with_suffix('.png')}")
    if args.figure_kind in {"aggregate", "both"}:
        out_prefix = args.out_dir / f"logistic_path_alignment_all_seeds_{args.strategy}"
        build_aggregate_figure(result, analysis, out_prefix, args.strategy, args.overlay_budget)
        print(f"wrote {out_prefix.with_suffix('.pdf')}")
        print(f"wrote {out_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
