"""Camera-ready cross-cell figures combining 4 CIFAR-10 runs:
(partial-FT, full-FT) x (clean, noisy 40%).

Produces:
  cv_grid_acc_vs_budget.{pdf,png}      - 2x2 panel, acc vs k for each cell
  cv_noise_robustness_drop.{pdf,png}   - bar chart: clean-noisy accuracy drop per method
  cv_selected_quality_noisy_fullft.*   - selected clean/target fractions at k=500
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

CELLS = {
    ("partial", "clean"): ROOT / "local_clean.json",
    ("partial", "noisy"): ROOT / "local_full.json",
    ("full", "clean"):    ROOT / "external_results/cv_cifar10_fullft_clean_5seed/results.json",
    ("full", "noisy"):    ROOT / "external_results/cv_cifar10_fullft_20260426_220513/results.json",
}

STYLE = {
    "Random":         dict(color="#888888", marker="o", linestyle=":",  zorder=1),
    "EmbedRetrieval": dict(color="#4E79A7", marker="s", linestyle="--", zorder=2),
    "EL2N":           dict(color="#76B7B2", marker="v", linestyle="--", zorder=2),
    "GraNd":          dict(color="#59A14F", marker="^", linestyle="--", zorder=2),
    "LESS":           dict(color="#D62728", marker="D", linestyle="-",  zorder=3, linewidth=2.0),
    "TACS":           dict(color="#000000", marker="*", linestyle="-",  zorder=4, linewidth=2.4, markersize=10),
}


def load(p):
    return json.load(open(p))["summary"]


def plot_grid(out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharey=True)
    cell_titles = {
        ("partial", "clean"): "Partial-FT (layer4+fc) — clean pool",
        ("partial", "noisy"): "Partial-FT (layer4+fc) — 40% label noise",
        ("full", "clean"):    "Full-FT (ResNet-18) — clean pool",
        ("full", "noisy"):    "Full-FT (ResNet-18) — 40% label noise",
    }
    layout = [[("full", "clean"), ("full", "noisy")],
              [("partial", "clean"), ("partial", "noisy")]]
    for r in range(2):
        for c in range(2):
            ax = axes[r][c]
            cell = layout[r][c]
            summary = load(CELLS[cell])
            methods = [m for m in STYLE if m in summary]
            for m in methods:
                per_k = summary[m]["k"]
                ks = sorted(int(k) for k in per_k)
                means = [per_k[str(k)]["acc_mean"] for k in ks]
                stds = [per_k[str(k)]["acc_std"] for k in ks]
                ax.errorbar(ks, means, yerr=stds, label=m, capsize=3, **STYLE[m])
            ax.set_title(cell_titles[cell], fontsize=11)
            ax.grid(alpha=0.3)
            ax.set_ylim(-2, 88)
            if r == 1:
                ax.set_xlabel("Selected subset size (k)")
            if c == 0:
                ax.set_ylabel("Target binary test acc (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.02), fontsize=10, framealpha=0.95)
    fig.suptitle("CIFAR-10 cat-vs-dog retrieval: accuracy vs. selected subset size",
                 fontsize=13, y=1.00)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}.{{png,pdf}}")


def plot_noise_drop(out_path, k_target=500):
    """Bar chart: clean - noisy accuracy drop at fixed k, partial-FT and full-FT side-by-side."""
    methods = list(STYLE.keys())
    drops = {scope: {} for scope in ["partial", "full"]}
    for scope in ["partial", "full"]:
        clean = load(CELLS[(scope, "clean")])
        noisy = load(CELLS[(scope, "noisy")])
        for m in methods:
            if m in clean and m in noisy and str(k_target) in clean[m]["k"] and str(k_target) in noisy[m]["k"]:
                drops[scope][m] = clean[m]["k"][str(k_target)]["acc_mean"] - noisy[m]["k"][str(k_target)]["acc_mean"]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(methods))
    w = 0.38
    p_vals = [drops["partial"].get(m, np.nan) for m in methods]
    f_vals = [drops["full"].get(m, np.nan) for m in methods]
    ax.bar(x - w/2, p_vals, w, label="Partial-FT", color="#9ecae1", edgecolor="black")
    ax.bar(x + w/2, f_vals, w, label="Full-FT",    color="#3182bd", edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    for idx, m in enumerate(methods):
        if m in {"TACS", "LESS"} and np.isfinite(f_vals[idx]):
            ax.text(
                idx + w / 2,
                f_vals[idx] + (1.0 if f_vals[idx] >= 0 else -2.0),
                f"{f_vals[idx]:+.1f}",
                ha="center",
                va="bottom" if f_vals[idx] >= 0 else "top",
                fontsize=9,
                fontweight="bold" if m == "TACS" else "normal",
            )
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel(f"Accuracy degradation (clean − noisy), pp at k={k_target}")
    ax.set_title("Noise degradation at k=500 (lower is better)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}.{{png,pdf}}")


def plot_selected_quality(out_path, k_target=500):
    """Selected-set diagnostics for the noisy full-FT run."""
    summary = load(CELLS[("full", "noisy")])
    methods = [m for m in STYLE if m in summary and str(k_target) in summary[m]["k"]]
    clean_vals = [summary[m]["k"][str(k_target)]["clean_frac_mean"] * 100 for m in methods]
    target_vals = [summary[m]["k"][str(k_target)]["target_frac_mean"] * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(methods))
    w = 0.38
    ax.bar(x - w / 2, clean_vals, w, label="Clean-label fraction", color="#B7B7B7", edgecolor="black")
    ax.bar(x + w / 2, target_vals, w, label="Target-distribution fraction", color="#F28E2B", edgecolor="black")
    for idx, m in enumerate(methods):
        if m in {"TACS", "LESS"}:
            ax.text(idx - w / 2, clean_vals[idx] + 1.2, f"{clean_vals[idx]:.0f}", ha="center", fontsize=9)
            ax.text(idx + w / 2, target_vals[idx] + 1.2, f"{target_vals[idx]:.0f}", ha="center", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=18, ha="right")
    ax.set_ylabel(f"Selected examples (%), noisy full-FT at k={k_target}")
    ax.set_title("Selected-set composition under 40% label noise")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}.{{png,pdf}}")


if __name__ == "__main__":
    outdir = ROOT / "plots_camera_ready"
    outdir.mkdir(exist_ok=True)
    plot_grid(outdir / "cv_grid_acc_vs_budget")
    plot_noise_drop(outdir / "cv_noise_robustness_drop", k_target=500)
    plot_selected_quality(outdir / "cv_selected_quality_noisy_fullft", k_target=500)
