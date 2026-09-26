#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CORE_METHODS = ["TACS", "ToV_max_improv", "LESS_multi", "MaxUncert", "Random", "Oracle_source"]
EXACT_FOCUSED_METHODS = ["TACS", "ToV_max_improv", "LESS_multi"]
METHOD_STYLE = {
    "TACS": {"label": "TACS", "color": "#c62828"},
    "ToV_max_improv": {"label": "ToV", "color": "#2ca02c"},
    "LESS_multi": {"label": "LESS", "color": "#1f77b4"},
    "MaxUncert": {"label": "Max Uncert.", "color": "#9467bd"},
    "Random": {"label": "Random", "color": "#9e9e9e"},
    "Oracle_source": {"label": "Oracle", "color": "#ff7f0e"},
}
STRATEGIES = ["random_from_top", "score_plus_random", "score_only"]
STRATEGY_LABELS = {
    "random_from_top": "Rand-frm-top",
    "score_plus_random": "Score+Random",
    "score_only": "Score-only",
}
PANEL_LABELS = ["A", "B", "C", "D", "E"]
SINGLE_PANEL_SIZE = (3.2, 2.8)


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def compute_ylim(summary: dict, budgets: list[int], methods: list[str], strategies: list[str], pad: float = 0.12) -> tuple[float, float]:
    vals = []
    for strategy in strategies:
        for method in methods:
            vals.extend(summary[strategy][method]["k"][str(k)]["classification_error_mean"] for k in budgets)
    y_min = min(vals)
    y_max = max(vals)
    span = max(y_max - y_min, 1e-4)
    margin = pad * span
    return y_min - margin, y_max + margin


def plot_exact_panel(
    ax: plt.Axes,
    summary: dict,
    budgets: list[int],
    strategy: str,
    panel_label: str,
    methods: list[str],
    *,
    compact: bool = False,
) -> None:
    for method in methods:
        style = METHOD_STYLE[method]
        color = style["color"]
        means = np.asarray(
            [summary[strategy][method]["k"][str(k)]["classification_error_mean"] for k in budgets],
            dtype=np.float64,
        )
        stds = np.asarray(
            [summary[strategy][method]["k"][str(k)]["classification_error_std"] for k in budgets],
            dtype=np.float64,
        )
        ax.plot(
            budgets,
            means,
            color=color,
            linewidth=2.5 if method == "TACS" else 1.9,
            marker="o",
            markersize=5.8 if method == "TACS" else 4.6,
            markerfacecolor="white" if method != "TACS" else color,
            markeredgewidth=1.2,
            label=style["label"],
        )
        ax.fill_between(budgets, means - stds, means + stds, color=color, alpha=0.10)

    ax.set_title(STRATEGY_LABELS[strategy])
    ax.set_xscale("log", base=2)
    if compact:
        ax.set_xticks([128, 512, 2048, 8192])
        ax.set_xticklabels(["128", "512", "2k", "8k"])
    else:
        ax.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
        ax.set_xticklabels(["128", "256", "512", "1k", "2k", "4k", "8k"])
    ax.set_xlabel("Final subset size n")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.9, alpha=0.35, color="#b8b8b8")
    add_panel_label(ax, panel_label)


def plot_exact_curves(data: dict, out_prefix: Path) -> None:
    budgets = [int(k) for k in data["config"]["budgets"]]
    summary = data["summary"]
    y_low, y_high = compute_ylim(summary, budgets, EXACT_FOCUSED_METHODS, STRATEGIES, pad=0.14)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), sharey=True)
    legend_handles = []
    legend_labels = []

    for ax, strategy, panel_label in zip(axes, STRATEGIES, PANEL_LABELS[:3]):
        plot_exact_panel(ax, summary, budgets, strategy, panel_label, EXACT_FOCUSED_METHODS)
        for method in EXACT_FOCUSED_METHODS:
            style = METHOD_STYLE[method]
            line = ax.plot([], [], color=style["color"], linewidth=2.5 if method == "TACS" else 1.9)[0]
            if strategy == STRATEGIES[0]:
                legend_handles.append(line)
                legend_labels.append(style["label"])
    axes[0].set_ylabel("Classification error")
    axes[0].set_ylim(y_low, y_high)

    fig.suptitle(
        "Exact ToV-style logistic mixture (focused view, m_val = 1024)",
        fontsize=15,
        fontweight="bold",
        y=1.10,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    for strategy, panel_label in zip(STRATEGIES, PANEL_LABELS[:3]):
        single_fig, single_ax = plt.subplots(1, 1, figsize=SINGLE_PANEL_SIZE)
        plot_exact_panel(single_ax, summary, budgets, strategy, "", EXACT_FOCUSED_METHODS, compact=True)
        single_ax.set_ylabel("Classification error")
        single_ax.set_ylim(y_low, y_high)
        style_compact_axis(single_ax)
        handles, labels = single_ax.get_legend_handles_labels()
        single_ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=7.0,
            handlelength=1.8,
            borderaxespad=0.0,
            labelspacing=0.35,
        )
        single_fig.subplots_adjust(left=0.24, right=0.68, bottom=0.23, top=0.84)
        out_path = out_prefix.parent / f"{out_prefix.name}_{strategy}"
        single_fig.savefig(out_path.with_suffix(".pdf"))
        single_fig.savefig(out_path.with_suffix(".png"))
        plt.close(single_fig)


def plot_exact_fullrange_curves(data: dict, out_prefix: Path) -> None:
    budgets = [int(k) for k in data["config"]["budgets"]]
    summary = data["summary"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), sharey=True)
    legend_handles = []
    legend_labels = []

    for ax, strategy, panel_label in zip(axes, STRATEGIES, PANEL_LABELS[:3]):
        plot_exact_panel(ax, summary, budgets, strategy, panel_label, CORE_METHODS)
        for method in CORE_METHODS:
            style = METHOD_STYLE[method]
            line = ax.plot([], [], color=style["color"], linewidth=2.5 if method == "TACS" else 1.9)[0]
            if strategy == STRATEGIES[0]:
                legend_handles.append(line)
                legend_labels.append(style["label"])
    axes[0].set_ylabel("Classification error")
    axes[0].set_ylim(0.30, 0.52)

    fig.suptitle(
        "Exact ToV-style logistic mixture (full range, m_val = 1024)",
        fontsize=15,
        fontweight="bold",
        y=1.10,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_lowshot_panel(
    ax: plt.Axes,
    data_map: dict[int, dict],
    val_sizes: list[int],
    main_budget: str,
    strategy: str,
    panel_label: str,
) -> None:
    methods = ["TACS", "ToV_max_improv", "LESS_multi", "Random", "Oracle_source"]
    for method in methods:
        style = METHOD_STYLE[method]
        color = style["color"]
        means = np.asarray(
            [
                data_map[mv]["summary"][strategy][method]["k"][main_budget]["classification_error_mean"]
                for mv in val_sizes
            ],
            dtype=np.float64,
        )
        stds = np.asarray(
            [
                data_map[mv]["summary"][strategy][method]["k"][main_budget]["classification_error_std"]
                for mv in val_sizes
            ],
            dtype=np.float64,
        )
        ax.plot(
            val_sizes,
            means,
            color=color,
            linewidth=2.5 if method == "TACS" else 1.9,
            marker="o",
            markersize=5.8 if method == "TACS" else 4.6,
            markerfacecolor="white" if method != "TACS" else color,
            markeredgewidth=1.2,
            label=style["label"],
        )
        ax.fill_between(val_sizes, means - stds, means + stds, color=color, alpha=0.10)

    ax.set_xscale("log", base=2)
    ax.set_xticks(val_sizes)
    ax.set_xticklabels([str(v) for v in val_sizes])
    ax.set_xlabel("Validation set size m_val")
    ax.set_title(f"{STRATEGY_LABELS[strategy]} at n={main_budget}")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.25)
    add_panel_label(ax, panel_label)


def plot_lowshot_trend(data_map: dict[int, dict], out_prefix: Path) -> None:
    val_sizes = sorted(data_map)
    main_budget = str(data_map[val_sizes[0]]["config"]["budgets"][-1])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, strategy, panel_label in zip(axes, ["score_plus_random", "score_only"], PANEL_LABELS[3:5]):
        plot_lowshot_panel(ax, data_map, val_sizes, main_budget, strategy, panel_label)
    axes[0].set_ylabel("Classification error")
    axes[0].set_ylim(0.30, 0.48)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Low-shot extension at n = 8192",
        fontsize=15,
        fontweight="bold",
        y=1.10,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    for strategy, panel_label in zip(["score_plus_random", "score_only"], PANEL_LABELS[3:5]):
        single_fig, single_ax = plt.subplots(1, 1, figsize=(5.1, 4.3))
        plot_lowshot_panel(single_ax, data_map, val_sizes, main_budget, strategy, "")
        single_ax.set_ylabel("Classification error")
        single_ax.set_ylim(0.30, 0.48)
        single_fig.tight_layout()
        single_fig.savefig((out_prefix.parent / f"{out_prefix.name}_{strategy}").with_suffix(".pdf"), bbox_inches="tight")
        plt.close(single_fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot conclusion figures for ToV-style logistic experiments.")
    parser.add_argument("--exact-json", type=Path, required=True, help="Exact ToV-style run with m_val=1024.")
    parser.add_argument("--lowshot-jsons", type=Path, nargs="*", default=[], help="Additional low-shot JSON files.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().with_name("plots_tov_style"),
        help="Directory to write figures.",
    )
    args = parser.parse_args()

    style_matplotlib()
    exact = load_json(args.exact_json.resolve())
    plot_exact_curves(exact, args.out_dir.resolve() / "logistic_tov_style_exact")
    plot_exact_fullrange_curves(exact, args.out_dir.resolve() / "logistic_tov_style_exact_fullrange")

    data_map = {int(exact["config"]["val_size"]): exact}
    for path in args.lowshot_jsons:
        data = load_json(path.resolve())
        data_map[int(data["config"]["val_size"])] = data
    if len(data_map) >= 2:
        plot_lowshot_trend(data_map, args.out_dir.resolve() / "logistic_tov_style_lowshot")

    print(f"out_dir={args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
