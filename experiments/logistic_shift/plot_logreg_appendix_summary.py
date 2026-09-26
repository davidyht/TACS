"""Replacement for tab:logreg_appendix_summary in Validation_Warmup/sections/appendix.tex.

Two-panel figure: (a) balanced-mixture target classification error vs k (broken y-axis
so the TACS/LESS/ToV cluster and the Random band are both readable),
(b) rare-target test accuracy vs k. Mean +/- std over 10 seeds.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import numpy as np

BAL_PATH = ROOT / "logistic_tov_style_comparison_full_corrected.json"
RARE_PATH = ROOT / "logistic_shift_framework_full.json"
OUT_DIR = ROOT.parent.parent.parent / "Validation_Warmup" / "fig"
OUT_STEM = OUT_DIR / "logreg_appendix_acc_vs_budget"

STYLE = {
    "TACS":   dict(color="#000000", marker="*", linestyle="-",  linewidth=2.5, markersize=9.5, zorder=4),
    "LESS":   dict(color="#E02020", marker="D", linestyle="-",  linewidth=2.0, markersize=5.6, zorder=3),
    "ToV":    dict(color="#4C78A8", marker="s", linestyle="--", linewidth=1.8, markersize=5.2, zorder=2),
    "Random": dict(color="#8A8A8A", marker="o", linestyle=":",  linewidth=1.7, markersize=5.0, zorder=1),
}

BAL_METHODS = {
    "TACS":   "TACS",
    "LESS":   "LESS_multi",
    "ToV":    "ToV_max_improv",
    "Random": "Random",
}
RARE_METHODS = {
    "TACS":   "TACS",
    "LESS":   "LESS_tracein",
    "ToV":    "ToV_1step",
    "Random": "Random",
}


def _plot_balanced_lines(ax):
    summary = json.load(open(BAL_PATH))["summary"]["score_only"]
    for label, key in BAL_METHODS.items():
        per_k = summary[key]["k"]
        ks = sorted(int(k) for k in per_k)
        means = np.array([per_k[str(k)]["classification_error_mean"] for k in ks])
        stds = np.array([per_k[str(k)]["classification_error_std"] for k in ks])
        ax.plot(ks, means, label=label, **STYLE[label])
        ax.fill_between(ks, means - stds, means + stds, color=STYLE[label]["color"], alpha=0.09, linewidth=0)


def _plot_rare(ax):
    summary = json.load(open(RARE_PATH))["summary"]
    for label, key in RARE_METHODS.items():
        per_k = summary[key]["performance"]["k"]
        ks = sorted(int(k) for k in per_k)
        means = np.array([per_k[str(k)]["target_test_acc_mean"] for k in ks])
        stds = np.array([per_k[str(k)]["target_test_acc_std"] for k in ks])
        ax.plot(ks, means, label=label, **STYLE[label])
        ax.fill_between(ks, means - stds, means + stds, color=STYLE[label]["color"], alpha=0.09, linewidth=0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.7, 3.2))
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.24, left=0.105, right=0.995, top=0.86, bottom=0.28)

    # Panel (a) — broken y-axis: top (Random band), bottom (TACS/LESS/ToV cluster)
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0, 0],
        height_ratios=[0.35, 1.0], hspace=0.06,
    )
    ax_top = fig.add_subplot(inner[0, 0])
    ax_bot = fig.add_subplot(inner[1, 0], sharex=ax_top)
    for a in (ax_top, ax_bot):
        _plot_balanced_lines(a)
        a.set_xscale("log", base=2)
        a.grid(axis="y", color="#D8D8D8", linewidth=0.7)
        a.grid(axis="x", visible=False)
        a.spines["right"].set_visible(False)
        a.tick_params(axis="both", labelsize=9)

    ax_top.set_ylim(0.370, 0.430)
    ax_bot.set_ylim(0.328, 0.346)
    ax_top.set_yticks([0.38, 0.40, 0.42])
    ax_bot.set_yticks([0.330, 0.335, 0.340, 0.345])
    ax_top.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_bot.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_top.set_title("(a) Balanced mixture", fontsize=11, pad=5)
    ax_bot.set_xlabel(r"Selection budget $k$", fontsize=10)
    ax_bot.set_xticks([128, 512, 2048, 8192])
    ax_bot.set_xticklabels(["128", "512", "2k", "8k"])

    d = 0.012
    kw = dict(color="k", clip_on=False, lw=0.9)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_bot.plot((-d, +d), (1 - d * 0.35, 1 + d * 0.35), transform=ax_bot.transAxes, **kw)
    ax_bot.plot((1 - d, 1 + d), (1 - d * 0.35, 1 + d * 0.35), transform=ax_bot.transAxes, **kw)

    fig.text(
        0.025,
        outer[0, 0].get_position(fig).y0 + outer[0, 0].get_position(fig).height / 2,
        "Target classification error",
        rotation=90, va="center", ha="center", fontsize=10,
    )

    # Panel (b)
    ax_b = fig.add_subplot(outer[0, 1])
    _plot_rare(ax_b)
    ax_b.set_xscale("log", base=2)
    ax_b.set_xlabel(r"Selection budget $k$", fontsize=10)
    ax_b.set_ylabel("Target test accuracy", fontsize=10)
    ax_b.set_title("(b) Rare target", fontsize=11, pad=5)
    ax_b.set_xticks([50, 100, 200, 400])
    ax_b.set_xticklabels(["50", "100", "200", "400"])
    ax_b.set_ylim(0.62, 0.965)
    ax_b.set_yticks([0.65, 0.75, 0.85, 0.95])
    ax_b.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_b.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    ax_b.grid(axis="x", visible=False)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.tick_params(axis="both", labelsize=9)

    handles, labels = ax_bot.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.03),
        fontsize=9.2,
        frameon=True,
        framealpha=0.95,
        borderpad=0.4,
        handlelength=2.1,
        columnspacing=1.25,
    )

    fig.savefig(str(OUT_STEM) + ".pdf", bbox_inches="tight")
    fig.savefig(str(OUT_STEM) + ".png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT_STEM}.{{pdf,png}}")


if __name__ == "__main__":
    main()
