"""Polished CIFAR-10 appendix budget figures.

This script regenerates the CV appendix figures used as Figure 7/8 in the
paper from the saved experiment summaries.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments" / "tacs_local" / "cv_cifar10_noisy"
OUT = Path(os.environ.get("CV_FIG_OUT", REPO / "Validation_Warmup" / "fig"))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

CELLS = {
    ("partial", "clean"): ROOT / "local_clean.json",
    ("partial", "noisy"): ROOT / "local_full.json",
    ("full", "clean"): ROOT / "vista_results" / "cv_cifar10_fullft_clean_5seed" / "results.json",
    ("full", "noisy"): ROOT / "vista_results" / "cv_cifar10_fullft_20260426_220513" / "results.json",
}

BUDGETS = [100, 250, 500, 1000]

# The saved full-FT run uses cosine-normalized gradient alignment (LESS), while
# the partial-FT run uses the raw inner product (TracIn). Keep their names distinct.
FULL_FT_LABELS = {"LESS": "LESS"}

STYLE = {
    "Random": {
        "label": "Random",
        "color": "#8A8A8A",
        "marker": "o",
        "linestyle": ":",
        "linewidth": 1.7,
        "markersize": 5.4,
        "alpha": 0.78,
        "zorder": 1,
    },
    "EmbedRetrieval": {
        "label": "Embed",
        "color": "#4C78A8",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.7,
        "markersize": 5.2,
        "alpha": 0.78,
        "zorder": 2,
    },
    "EL2N": {
        "label": "EL2N",
        "color": "#72B7B2",
        "marker": "v",
        "linestyle": "--",
        "linewidth": 1.7,
        "markersize": 5.4,
        "alpha": 0.78,
        "zorder": 2,
    },
    "GraNd": {
        "label": "GraNd",
        "color": "#54A24B",
        "marker": "^",
        "linestyle": "--",
        "linewidth": 1.7,
        "markersize": 5.4,
        "alpha": 0.78,
        "zorder": 2,
    },
    "LESS": {
        "label": "TracIn",
        "color": "#E02020",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 2.2,
        "markersize": 5.8,
        "alpha": 0.98,
        "zorder": 4,
    },
    "TACS": {
        "label": "TACS",
        "color": "#000000",
        "marker": "*",
        "linestyle": "-",
        "linewidth": 2.8,
        "markersize": 9.5,
        "alpha": 1.0,
        "zorder": 5,
    },
    "GroupedTACS": {
        "label": "TACS-G",
        "color": "#B23CB2",
        "marker": "P",
        "linestyle": "-",
        "linewidth": 2.4,
        "markersize": 7.5,
        "alpha": 1.0,
        "zorder": 6,
    },
}


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())["summary"]


def method_order(summary: dict) -> list[str]:
    return [method for method in STYLE if method in summary]


def panel_budgets(summary: dict) -> list[int]:
    seen: set[int] = set()
    for method in method_order(summary):
        seen.update(int(k) for k in summary[method]["k"])
    return [k for k in BUDGETS if k in seen]


def plot_cell(
    ax: plt.Axes,
    summary: dict,
    *,
    title: str | None = None,
    ylabel: bool = False,
    xlabel: bool = False,
    legend: bool = False,
    label_overrides: dict[str, str] | None = None,
) -> None:
    budgets = panel_budgets(summary)
    xpos = {k: i for i, k in enumerate(budgets)}
    handles = []
    labels = []
    for method in method_order(summary):
        per_k = summary[method]["k"]
        ks = [k for k in budgets if str(k) in per_k]
        means = np.array([per_k[str(k)]["acc_mean"] for k in ks], dtype=float)
        stds = np.array([per_k[str(k)]["acc_std"] for k in ks], dtype=float)
        cfg = STYLE[method]
        label = (label_overrides or {}).get(method, cfg["label"])
        handle = ax.errorbar(
            ks,
            means,
            yerr=stds,
            label=label,
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=cfg["linestyle"],
            linewidth=cfg["linewidth"],
            markersize=cfg["markersize"],
            alpha=cfg["alpha"],
            zorder=cfg["zorder"],
            capsize=2.5,
            capthick=1.0,
            elinewidth=1.0 if method in {"TACS", "LESS"} else 0.85,
        )
        handles.append(handle)
        labels.append(label)

    if title:
        ax.set_title(title, fontsize=11.5, pad=6)
    ax.set_ylim(0, 86)
    ax.set_yticks([0, 20, 40, 60, 80])
    span = budgets[-1] - budgets[0]
    pad = max(20, 0.04 * span)
    ax.set_xlim(budgets[0] - pad, budgets[-1] + pad)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(k) for k in budgets])
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ylabel:
        ax.set_ylabel("Target test accuracy (%)", fontsize=11)
    if xlabel:
        ax.set_xlabel("Selected subset size k", fontsize=11)
    if legend:
        ax.legend(
            loc="lower right",
            ncol=2,
            fontsize=8.6,
            frameon=True,
            framealpha=0.92,
            borderpad=0.45,
            handlelength=2.0,
            columnspacing=1.0,
        )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=240)
    plt.close(fig)


def plot_pair(
    cells: list[tuple[str, str]],
    titles: list[str],
    stem: str,
    label_overrides: dict[str, str] | None = None,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.25), sharey=True)
    for col, (cell, title) in enumerate(zip(cells, titles)):
        plot_cell(
            axes[col],
            load_summary(CELLS[cell]),
            title=title,
            ylabel=col == 0,
            xlabel=True,
            label_overrides=label_overrides,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9.2,
        frameon=True,
        framealpha=0.94,
        borderpad=0.45,
        handlelength=2.2,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.095, right=0.995, top=0.85, bottom=0.28, wspace=0.10)
    save(fig, stem)


def plot_grid() -> None:
    plot_pair(
        [("full", "clean"), ("full", "noisy")],
        ["(a) Clean pool", "(b) 40% noisy pool"],
        "cv_grid_acc_vs_budget",
        label_overrides=FULL_FT_LABELS,
    )


def plot_partial_panels() -> None:
    plot_pair(
        [("partial", "clean"), ("partial", "noisy")],
        ["(a) Clean pool", "(b) 40% noisy pool"],
        "cv_partialft_acc_vs_budget",
    )
    for pool, stem, legend in [
        ("clean", "cv_partialft_clean_acc_vs_budget", False),
        ("noisy", "cv_partialft_noisy_acc_vs_budget", True),
    ]:
        title = "Clean pool" if pool == "clean" else "40% noisy pool"
        fig, ax = plt.subplots(figsize=(4.25, 3.05))
        plot_cell(
            ax,
            load_summary(CELLS[("partial", pool)]),
            title=title,
            ylabel=pool == "clean",
            xlabel=True,
            legend=legend,
        )
        fig.subplots_adjust(left=0.17 if pool == "clean" else 0.08, right=0.99, top=0.88, bottom=0.17)
        save(fig, stem)


def main() -> None:
    plot_grid()
    plot_partial_panels()


if __name__ == "__main__":
    main()
