"""Polish appendix figures for TACS length-normalization diagnostics.

The heavy analysis writes CSV files under analysis_outputs/.  This script only
replots those cached diagnostics into paper-facing PDFs.
"""

from __future__ import annotations

from pathlib import Path

import os

REPO = Path(__file__).resolve().parents[2]
IN_DIR = REPO / "analysis_outputs" / "tacs_length_normalization_l32_3b"
OUT_DIR = REPO / "Validation_Warmup" / "fig"
os.environ.setdefault("MPLCONFIGDIR", str(IN_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

TASKS = ["tydiqa", "mmlu", "bbh"]
SOURCES = ["flan_v2", "cot", "oasst1", "dolly"]

TASK_LABELS = {"tydiqa": "TyDiQA", "mmlu": "MMLU", "bbh": "BBH"}
SOURCE_LABELS = {"flan_v2": "Flan V2", "cot": "CoT", "oasst1": "OASST1", "dolly": "Dolly"}

TASK_COLORS = {"tydiqa": "#009E73", "mmlu": "#D55E00", "bbh": "#5E60B8"}
SOURCE_MARKERS = {"flan_v2": "o", "cot": "s", "oasst1": "^", "dolly": "D"}

POOL_COLOR = "#CFCFCF"
RAW_COLOR = "#D55E00"
NORM_COLOR = "#0072B2"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight", dpi=240)
    plt.close(fig)


def plot_scatter(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 4.35))

    lo = 20
    hi = 380
    ax.plot([lo, hi], [lo, hi], color="#333333", linestyle="--", linewidth=0.9, zorder=0)
    ax.text(338, 354, "equal length", fontsize=8, rotation=41, color="#444444", ha="center")

    for _, row in summary.iterrows():
        task = row["task"]
        source = row["source"]
        ax.scatter(
            row["raw_mean_words"],
            row["norm_mean_words"],
            s=66,
            color=TASK_COLORS[task],
            marker=SOURCE_MARKERS[source],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.95,
            zorder=3,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Raw loss gap: mean selected words", fontsize=10)
    ax.set_ylabel("Normalized score: mean selected words", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(color="#E0E0E0", linewidth=0.65, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    task_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=TASK_COLORS[task],
            markerfacecolor=TASK_COLORS[task],
            markeredgecolor="white",
            markersize=7,
            label=TASK_LABELS[task],
        )
        for task in TASKS
    ]
    source_handles = [
        Line2D(
            [0],
            [0],
            marker=SOURCE_MARKERS[source],
            linestyle="",
            color="#333333",
            markerfacecolor="#333333",
            markeredgecolor="white",
            markersize=7,
            label=SOURCE_LABELS[source],
        )
        for source in SOURCES
    ]

    leg1 = ax.legend(
        handles=task_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        framealpha=0.93,
        fontsize=8.5,
        borderpad=0.35,
        labelspacing=0.3,
        title="Task",
        title_fontsize=8.5,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=source_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        framealpha=0.93,
        fontsize=8.2,
        borderpad=0.35,
        labelspacing=0.28,
        title="Source",
        title_fontsize=8.5,
    )

    fig.tight_layout(pad=0.4)
    save(fig, "length_normalization_scatter")


def cell_bins(pool_vals: np.ndarray) -> np.ndarray:
    hi = max(80.0, float(np.percentile(pool_vals, 98.5)))
    return np.linspace(0.0, hi, 22)


def plot_hist_grid(dist: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        len(TASKS),
        len(SOURCES),
        figsize=(7.45, 4.55),
        sharex=False,
        sharey=False,
    )

    for i, task in enumerate(TASKS):
        for j, source in enumerate(SOURCES):
            ax = axes[i, j]
            sub = dist[(dist["task"] == task) & (dist["source"] == source)]
            pool = sub[sub["selection"] == "pool"]["total_words"].to_numpy()
            raw = sub[sub["selection"] == "raw"]["total_words"].to_numpy()
            norm = sub[sub["selection"] == "norm"]["total_words"].to_numpy()
            bins = cell_bins(pool)

            ax.hist(pool, bins=bins, density=True, histtype="stepfilled", color=POOL_COLOR, alpha=0.42, linewidth=0)
            ax.hist(raw, bins=bins, density=True, histtype="step", color=RAW_COLOR, linewidth=1.25)
            ax.hist(norm, bins=bins, density=True, histtype="step", color=NORM_COLOR, linewidth=1.35)

            for vals, color, style in [(pool, "#9A9A9A", ":"), (raw, RAW_COLOR, "--"), (norm, NORM_COLOR, "--")]:
                ax.axvline(np.mean(vals), color=color, linestyle=style, linewidth=0.9, alpha=0.95)

            if i == 0:
                ax.set_title(SOURCE_LABELS[source], fontsize=9.5, pad=5)
            if j == 0:
                ax.set_ylabel(TASK_LABELS[task], fontsize=9.5, rotation=0, labelpad=20, va="center")
            if i == len(TASKS) - 1:
                ax.set_xlabel("Total words", fontsize=8.1)
            else:
                ax.set_xticklabels([])

            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=6.9, length=2.5)
            ax.tick_params(axis="y", length=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#333333")
            ax.spines["bottom"].set_color("#333333")

    handles = [
        Patch(facecolor=POOL_COLOR, edgecolor="none", alpha=0.42, label="Pool"),
        Line2D([0], [0], color=RAW_COLOR, linewidth=1.4, label="Raw top-5%"),
        Line2D([0], [0], color=NORM_COLOR, linewidth=1.5, label="Normalized top-5%"),
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=0.9, label="Mean"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        framealpha=0.95,
        fontsize=8.4,
        borderpad=0.4,
        handlelength=2.0,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(left=0.085, right=0.995, top=0.91, bottom=0.145, hspace=0.22, wspace=0.14)
    save(fig, "length_normalization_hist_grid")


def main() -> None:
    configure()
    summary = pd.read_csv(IN_DIR / "length_normalization_summary.csv")
    dist = pd.read_csv(IN_DIR / "length_normalization_distributions.csv")
    plot_scatter(summary)
    plot_hist_grid(dist)


if __name__ == "__main__":
    main()
