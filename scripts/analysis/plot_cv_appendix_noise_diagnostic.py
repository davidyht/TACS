#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiments" / "tacs_local" / "cv_cifar10_noisy"
OUT = REPO / "Validation_Warmup" / "fig" / "cv_fullft_noise_diagnostic"
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".mpl"))

import matplotlib.pyplot as plt


FULL_CLEAN = ROOT / "vista_results" / "cv_cifar10_fullft_clean_5seed" / "results.json"
FULL_NOISY = ROOT / "vista_results" / "cv_cifar10_fullft_20260426_220513" / "results.json"

STYLE = {
    "TACS": {"color": "#d62728", "marker": "o", "label": "TACS"},
    "LESS": {"color": "#1f77b4", "marker": "s", "label": "LESS"},
    "Random": {"color": "#777777", "marker": "^", "label": "Random"},
}


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def series(summary: dict, method: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_k = summary[method]["k"]
    ks = np.asarray(sorted(int(k) for k in per_k), dtype=int)
    means = np.asarray([per_k[str(k)][metric] for k in ks], dtype=float)
    std_key = metric.replace("_mean", "_std")
    stds = np.asarray([per_k[str(k)].get(std_key, 0.0) for k in ks], dtype=float)
    return ks, means, stds


def main() -> None:
    clean = load_summary(FULL_CLEAN)
    noisy = load_summary(FULL_NOISY)

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 7.0,
        "axes.titlesize": 7.8,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.05), gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    for method in ["TACS", "LESS"]:
        cfg = STYLE[method]
        ks, y_clean, s_clean = series(clean, method, "acc_mean")
        _, y_noisy, s_noisy = series(noisy, method, "acc_mean")
        ax.errorbar(ks, y_clean, yerr=s_clean, color=cfg["color"], marker=cfg["marker"],
                    linestyle="-", linewidth=1.35, markersize=3.6, capsize=2.0,
                    label=f"{cfg['label']} clean")
        ax.errorbar(ks, y_noisy, yerr=s_noisy, color=cfg["color"], marker=cfg["marker"],
                    linestyle="--", linewidth=1.15, markersize=3.4, capsize=2.0,
                    alpha=0.78, label=f"{cfg['label']} noisy")
    ax.set_title("Full-FT accuracy")
    ax.set_xlabel("subset size")
    ax.set_ylabel("target acc. (%)")
    ax.set_xticks(ks)
    ax.set_xticklabels(["100", "250", "500", "1k"])
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, loc="lower right", handlelength=1.5, labelspacing=0.15)

    for ax, metric, title, ylabel in [
        (axes[1], "clean_frac_mean", "Noisy pool: clean labels", "selected clean (%)"),
        (axes[2], "target_frac_mean", "Noisy pool: target class", "selected target (%)"),
    ]:
        for method in ["TACS", "LESS", "Random"]:
            cfg = STYLE[method]
            ks, vals, _ = series(noisy, method, metric)
            ax.plot(ks, 100.0 * vals, color=cfg["color"], marker=cfg["marker"],
                    linewidth=1.35 if method != "Random" else 1.05,
                    linestyle=":" if method == "Random" else "-",
                    markersize=3.5, label=cfg["label"])
        ax.set_title(title)
        ax.set_xlabel("subset size")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 105)
        ax.set_xticks(ks)
        ax.set_xticklabels(["100", "250", "500", "1k"])
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        if ax is axes[2]:
            ax.legend(frameon=False, loc="lower left", handlelength=1.4, labelspacing=0.15)

    for label, ax in zip(["a", "b", "c"], axes):
        ax.text(-0.18, 1.06, label, transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
